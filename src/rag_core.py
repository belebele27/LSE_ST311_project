
import torch
import clip
from tqdm import tqdm 

class RAG:
    def __init__(self, encoder, preprocess, client):
        self.client = client
        self.encoder = encoder
        self.preprocess = preprocess
        self.search_space = None
        self.corpus = None

    def build_search_space(self, corpus_path, device="mps", batch_size=32):
        with open(corpus_path, "r") as f:
            corpus = self._split_long_sentences(list(filter(lambda x : x != "", [line.strip() for line in f.readlines()])))
        self.corpus = corpus
        
        search_space = []
        
        print("building search space...")
        for i in tqdm(range(0, len(corpus), batch_size), desc="encoding text"):
            batch_chunks = corpus[i:i + batch_size]
            batch_chunks = [line for line in batch_chunks if line]
            if not batch_chunks:
                continue
                
            batch_tokens = clip.tokenize(batch_chunks).to(device)
            
            with torch.no_grad():
                batch_features = self.encoder.encode_text(batch_tokens)
                batch_features = batch_features / batch_features.norm(dim=1, keepdim=True)
                search_space.append(batch_features)
        
        if search_space:
            search_space = torch.cat(search_space, dim=0)
        else:
            search_space = torch.zeros((0, self.encoder.text_projection.shape[1]), device=device)

        self.search_space = search_space
        
        return search_space, corpus

    def _search_text_by_image(self, image, device="mps", top_k=5):

        if not isinstance(image, torch.Tensor):
            image_input = self.preprocess(image).unsqueeze(0).to(device)
        else:
            image_input = image.to(device)
        
        with torch.no_grad():
            image_features = self.encoder.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        similarity = image_features @ self.search_space.T
        
        top_values, top_indices = similarity.topk(min(top_k, len(self.corpus)))
        
        results = []
        for i, idx in enumerate(top_indices[0]):
            results.append((self.corpus[idx.item()], top_values[0][i].item()))
        
        return results

    def _split_long_sentences(self, sentences, max_words=50):
        result = []
        
        print("splitting long sentences into smaller chunks...")
        for sentence in tqdm(sentences, desc="splitting sentences"):
            words = sentence.split()
            
            if len(words) <= max_words:
                result.append(sentence)
            else:
                for i in range(0, len(words), max_words):
                    chunk = words[i:i + max_words]
                    result.append(" ".join(chunk))
        
        return result

    def forward(self, image, query, device="mps", top_k=1, debug=False):
        descriptions = "\n".join([elem[0] for elem in self._search_text_by_image(image, device=device, top_k=top_k)])
        if debug:
            print(f"Descriptions:\n {descriptions}")
            print(f"Query:\n {query}")
        
        prompt = """
            You are an image question answering assistant. Given descriptions about an image, 
            answer the following question concisely, ideally in a single lowercase word (as short as possible). 
            Only use information explicitly provided in the descriptions as the answer is in the descriptions.

        Descriptions: {}

        Question: {}

        Answer:
        """.format(descriptions, query)
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt]
        )
        return response.text

    def set_client(self, client):
        self.client = client
