# Vesta
Semantic image search in discord. Search all images in a sever with a text phrase!

## Install

`git clone https://github.com/Ednaordinary/Vesta/`

`cd Vesta`

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

`cd laion/bigG`

`git-lfs https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`

`mv CLIP-ViT-bigG-14-laion2B-39B-b160k 0_CLIPModel`

`mv preprocessor_config.json ./0_CLIPModel`

`cd ../..`

Then, create a ".env" with "DISCORD_TOKEN=" set to your token.

`python3 ./main.py`

The bot will start indexing all servers it is in straight away. Only embeddings are saved, so it shouldn't use too much storage. Your GPU will turn on a lot during this, as it downloads images as fast as possible and runs them in batches to the gpu.

## Showcase

Results in my server for "cat"
![image](https://github.com/Ednaordinary/Vesta/assets/88869424/271c5697-917c-4430-91a7-9343658fb548)
