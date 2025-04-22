from dotenv import load_dotenv
import os
from openai import OpenAI

# Carrega a variável OPENAI_API_KEY do .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ A chave OPENAI_API_KEY não foi encontrada no .env")
    exit()

# Testar se a chave funciona com uma chamada simples à OpenAI
try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Teste: você está funcionando?"}
        ]
    )
    print("✅ Conexão bem-sucedida!")
    print("Resposta da OpenAI:", response.choices[0].message.content.strip())

except Exception as e:
    print("❌ Erro ao conectar com a OpenAI:")
    print(e)
