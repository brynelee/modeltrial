from pydantic import BaseModel

class Person(BaseModel):
    name: str

p = Person(name="Tom")
print(p.json())

p = {"name": "Tom"}
p = Person(**p)
print(p.json())

p2 = Person.copy(p)
print(p2.json())