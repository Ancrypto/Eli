# Eli

You will need Python3 and you need to be able to import all these packages:

nltk
numpy
tflearn
tensorflow
random
json
pickle
os
colorama
shutil

Use pip3 to install them.

You can edit the intents.json file located at Eli/Brain/JSON/intents.json to add vocabulary to Eli.

-----------------------
Booting for first time.
-----------------------

Before you boot up main.py you need to open a python3 shell and import nltk, then you will need to run the command nltk.download('punkt').
When you've done that you can run main.py and a console will open.
Type "train" in the console and it will train Eli with all the vocabulary that you've added in the intents.json file.
After that finishes, run "chat" in the console and you will be able to chat with Eli!
