from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5",
    input="tell me a joke"
)

print(response.output_text)
