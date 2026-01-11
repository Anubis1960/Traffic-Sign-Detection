
# Traffic Sign Detection

A simple python script that detects traffic signs from videos and images using opencv and a custon tensorflow CNN


## Installation


Clone repository
```bash
  git clone git@github.com:Anubis1960/Traffic-Sign-Detection.git
```

Install dependencies
```bash
  pip install -r ./requirements.txt
```

Create the model
- The model is trained using [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/code?datasetId=82373&searchQuery=label) and [Flickr](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset/data) 



## Usage

```bash
  python3 main.py --input <path_to_input_file> --model <path_to_cnn_model> --output <path_to_output_file_for_video_analysis>
```
