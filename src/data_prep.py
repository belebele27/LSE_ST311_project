
from google.genai import types

def create_image_description(instance, client):
    prompt = generate_prompt(instance)
    image = instance["image"].save("../data/foo.png")
    with open("../data/foo.png", "rb") as f:
        image_bytes = f.read()
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ),
        prompt
        ]
    )
    return response.text

def generate_prompt(instance):
    question = instance["question"]
    answers = instance["answers"]
    answers = " ".join(answers)
    prompt = """ 
        You are an image describer. Your task is to describe the image.

        You will be given an image, a question, and some possible answers.

        You need to identify the most popular answer from the list of answers.

        Never answer the question directly. Instead, you MUST describe the image in a short paragraph (under 50 words) around one of the answers. 
        
        Your description must include the most popular answer  without making it obvious that you are answering the question.

        You need to make the answer stand out without adding special characters nor making it Capitalized.

        Focus solely on describing the objects and scene within the image. Do not mention the image itself or provide any introduction. Avoid using special characters.

        Question: {}
        Answer: {}
    """.format(question, answers)
    return prompt