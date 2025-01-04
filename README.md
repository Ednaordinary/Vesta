# Vesta

## Find anything on a discord server in 10 seconds



https://github.com/user-attachments/assets/b75c8449-a873-404f-b161-c38948e718a1



## Install

`git clone https://github.com/Ednaordinary/Vesta/`

`cd Vesta`

`python3 -m venv venv`

`source venv/bin/activate`

`pip install -r requirements.txt`

`cd laion/DCXL`

`git-lfs https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`

`mv CLIP-ViT-L-14-DataComp.XL-s13B-b90K 0_CLIPModel`

`mv preprocessor_config.json ./0_CLIPModel`

`cd ../..`

Then, create a ".env" with "DISCORD_TOKEN=" set to your token.

`python3 ./main.py`

The bot will start indexing all servers it is in straight away. Only embeddings are saved, so it shouldn't use too much storage. Your GPU will turn on a lot during this, as it downloads images as fast as possible and runs them in batches to the gpu. It only fetchs the messages and images it needs to, everything else is cached.
