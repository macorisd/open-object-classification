import ollama

res = ollama.chat(
   # model='llava:13b',
    model='llava:34b',
    messages=[
       {'role': 'user',
        'content': 'Describe this image.',
        'images':['./input_image.jpg'] 
        } 
    ]
)

print(res['message']['content'])