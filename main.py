import nextcord as discord
import sys
import time
import io
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
import threading
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as ST_util
from PIL import Image
import numpy as np
import shutil
from pathlib import Path
import torch
import requests

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
#model = SentenceTransformer('clip-ViT-L-14').to("cpu")
#model = SentenceTransformer('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k').to("cpu")
model = None
model_users = 0
model_queue = []
searchers = 0
download_queue = []

def download_thread(url, path):
    global model_queue
    content = requests.get(url).content
    img = Image.open(io.BytesIO(content))
    model_queue.append(tuple((img, path)))

def downloader():
    global download_queue
    while True:
        while download_queue == []:
            time.sleep(0.01)
        now = [x for x in download_queue]
        download_queue = []
        for url, path in now:
            threading.Thread(target=download_thread, args=[url, path]).start()

def encoder():
    global model
    global model_queue
    global searchers
    model = SentenceTransformer('laion/bigG', local_files_only=True, cache_folder="./", model_kwargs=dict(attn_implementation="flash_attention_2")).to("cpu")
    while True:
        while model_queue == []:
            if model.device.type == "cpu" and searchers <= 0:
                searchers = 0
                model.to('cpu')
            time.sleep(0.02)
        if model.device.type != "cuda": model.to("cuda")
        now = [x for x in model_queue]
        model_queue = []
        paths = []
        contents = []
        print("now:", len(now))
        for url, path in now:
            paths.append(path)
            contents.append(url)
            #try:
            #content = requests.get(url).content
            #img = Image.open(io.BytesIO(content))
            #contents.append(img)
            #except Exception as e:
            #    exc_type, exc_obj, exc_tb = sys.exc_info()
            #    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #    print(exc_type, fname, exc_tb.tb_lineno)
            #    print(repr(e))
            #    paths.pop(-1)
            #    pass
        print("contents", len(contents))
        if len(contents) != 0:
            embeds = model.encode(contents)
            for idx, embed in enumerate(embeds):
                np.save(paths[idx], embed)
                print("^"*len(embeds), flush=True, end='')
        #try:
        #    url = model_queue[0][0]
        #    path = model_queue[0][1]
        #    content = requests.get(url).content
        #    img = Image.open(io.BytesIO(content))
        #    embeds = model.encode(img)
        #    np.save(path, embeds)
        #    print("^", flush=True, end='')
        #except Exception as e:
        #    exc_type, exc_obj, exc_tb = sys.exc_info()
        #    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #    print(exc_type, fname, exc_tb.tb_lineno)
        #    print(repr(e))
        #    pass
        #model_queue.pop(0)

async def image_indexer(channel):
    global model
    global model_users
    global download_queue
    try:
        if os.path.exists("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt"):
            with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt", "r") as progress_lock:
                # If a channel was in progress, it cannot be recovered. Delete and start over.
                if progress_lock.readlines()[0][0] == "1":
                    shutil.rmtree("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/")
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id), exist_ok=True)
                else:
                    #Channel is already properly indexed. Stop now.
                    model_users -= 1
                    return
        with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt", "w") as progress_lock:
            # If a channel indexing is in progress and we try to search, we need to send an error to the user.
            progress_lock.write("1")
        async for message in channel.history(limit=None):
            print(".", end='', flush=True)
            if message.attachments and message.attachments != [] and message.author.id != client.user.id:
                if type(channel) == discord.TextChannel:
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + str(message.id), exist_ok=True)
                    for idx, attachment in enumerate(message.attachments):
                        if attachment.content_type in ["image/jpeg", "image/png"]:
                            print("x", flush=True, end='')
                            download_queue.append(tuple((attachment, "./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))                        
                elif type(channel) == discord.Thread:
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/threads", exist_ok=True)
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(channel.id), exist_ok=True)
                    for idx, attachment in enumerate(message.attachments):
                        if attachment.content_type in ["image/jpeg", "image/png"]:
                            print("x", flush=True, end='')
                            download_queue.append(tuple((attachment, "./index/" + str(channel.guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(channel.id) + "/" + str(idx) + ".npy")))
        with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt", "w") as progress_lock:
            progress_lock.write("0")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        pass #threading stuff is REALLY important to not break
    model_users -= 1

def call_read_channel(channel):
    #this function might not have much reason to exist
    #channel = client.get_channel(channel)
    if channel is not None:
        asyncio.run_coroutine_threadsafe(coro=image_indexer(channel), loop=client.loop)

def add_guild_instance(guild):
    #SHOULD BE CALLED IN A THREAD. otherwise, event loop blocks and it never completes
    global model_users
    if not os.path.isdir("./index/" + str(guild.id)):
        os.makedirs("./index/" + str(guild.id), exist_ok=True)
    for channel in guild.channels:
        if model_users > 3:
            while model_users > 3:
                time.sleep(0.1)
        model_users += 1
        if type(channel) == discord.TextChannel:
            os.makedirs("./index/" + str(guild.id) + "/" + str(channel.id), exist_ok=True)
            call_read_channel(channel)
        elif type(channel) == discord.ForumChannel:
            for thread in channel.threads:
                os.makedirs("./index/" + str(guild.id) + "/" + str(thread.parent.id) + "/threads/" + str(thread.id), exist_ok=True)
                call_read_channel(thread)
        elif type(channel) == discord.Thread:
            os.makedirs("./index/" + str(guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(channel.id), exist_ok=True)
            call_read_channel(channel)
        else:
            model_users -= 1
    while model_users > 0:
        time.sleep(0.01)
    print("\nDone with", guild.name)

@client.event
async def on_guild_join(guild):
    threading.Thread(target=add_guild_instance, args=[guild]).start()

@client.event
async def on_error(event, *args, **kwargs):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    print(repr(event))
    raise

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    os.makedirs("./index", exist_ok=True)
    for guild in client.guilds:
        threading.Thread(target=add_guild_instance, args=[guild]).start()
            

@client.slash_command(description="Search for an image with the description.")
async def find_image(
        interaction: discord.Interaction,
        terms: str = discord.SlashOption(
            name="terms",
            required=True,
            description="Terms to search for, separated by comma.",
        ),
):
    await interaction.response.defer()
    global searchers
    global model
    if model == None:
        await interaction.followup.send("The bot is still initializing.")
    else:
        terms = terms.split(",")
        terms = [x.strip() for x in terms]
        searchers += 1
        model.to("cuda")
        term_embeds = model.encode(terms)
        searchers -= 1
        paths = []
        embeds = []
        for path in Path('./index/' + str(interaction.guild.id)).rglob('*.npy'):
            paths.append(path)
            embeds.append(np.load(path))
        scores = ST_util.dot_score(np.array(term_embeds), np.array(embeds))
        values, idxs = torch.topk(scores, 10)
        images = []
        for idx in idxs[0]:
            if isinstance(idx.item(), int):
                images.append(paths[int(idx)])
        image_attachments = []
        for path in images:
            path = str(path)
            try:
                if "threads" in path:
                    channel = client.get_channel(int(path.split("/")[4]))
                    message = await channel.fetch_message(int(path.split("/")[5]))
                    number = int(path.split("/")[6].split(".")[0])
                    image_attachments.append(message.attachments[number])
                else:
                    channel = client.get_channel(int(path.split("/")[2]))
                    message = await channel.fetch_message(int(path.split("/")[3]))
                    number = int(path.split("/")[4].split(".")[0])
                    image_attachments.append(message.attachments[number])
            except Exception as e:
                print("failed to find!")
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                pass
        results = []
        for idx, attachment in enumerate(image_attachments):
            with io.BytesIO() as imagebn:
                await attachment.save(fp=imagebn)
                file_end = ""
                if attachment.content_type == "image/jpeg":
                    file_end = ".jpg"
                elif attachment.content_type == "image/png":
                    file_end = ".png"
                imagebn.seek(0)
                results.append(discord.File(fp=imagebn, filename=str(idx) + file_end))
        await interaction.followup.send("Here are " + str(len(results)) + " results.", files=results)

threading.Thread(target=downloader).start()
threading.Thread(target=encoder).start()
client.run(TOKEN)
