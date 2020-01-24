# Visual Question Answering

The aim of this challenge is to solve a visual question answering (VQA) problem on the proposed dataset, which contains synthetic scenes composed by several objects (see examples above) and corresponding questions about the existence of something in the scene (e.g., Is there a yellow thing?') or about counting (e.g., How many big objects are there?') . Given an image and a question, the goal is to provide the correct answer.
 
 ## Data Details
* Images 
	* **Image size**:  320x480 pixels
	* **Color space**: RGB
	* **File Format**: png
	* **Number of training images**: 69642
	* **Number of test images**: 2754

* Questions
	* **Two types of questions**: "exist" or "count"
	* **Number of training questions**: 259492
	* **Number of test questions**: 3000

*Answers (13 possible labels):
	* '0': 0
	* '1': 1
	* '10': 2
	* '2': 3
	* '3': 4
	* '4': 5
	* '5': 6
	* '6': 7
	* '7': 8
	* '8': 9
	* '9': 10
	* 'no': 11
	* 'yes': 12

## Dataset Structure
The dataset contains two main folders: train and test. Folder train contains images corresponding to training questions. Folder test contains images corresponding to test questions. The dataset contains two json files: *train_data.json* and *test_data.json*. Each json file contains a list of questions. You can use the following script to read them.

```
import json

with open('/PATH/TO/DATASET/SUBSET_data.json', 'r') as f:
      SUBSET_data = json.load(f)
f.close()

SUBSET_data = SUBSET_questions
```

Each question is a dictionary as the following
```
{
 'question': ...,
 'image_filename': ..., 
 'answer': ...
}
```

where 'question' is a sentence, e.g., 'How many red objects?', 'image_filename', is the filename of the image the question is referring to, 'answer' is the ground truth (one of {'0', '1', '10', ..., 'no', 'yes'}).

Test questions have an additional key that is a 'question_id' to uniquely identify your solution when submittin