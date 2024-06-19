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

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
model = SentenceTransformer('clip-ViT-L-14').to("cpu")
model_loading = False
model_users = 0


def model_loader():
    global model
    global model_loading
    if not model_loading:
        if model.device.type == 'cpu':
            model_loading = True
            model = model.to('cuda')
            model_loading = False
    else:
        while model.device.type == 'cpu':
            time.sleep(0.01)

async def image_indexer(channel):
    global model
    global model_users
    global model_loading
    print("image_indexer called. current stats:")
    print("model_loading", model_loading)
    print("model_users", model_users)
    print("model device type ", model.device.type)
    try:
        if os.path.exists("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt"):
            print("Channel exists")
            with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt", "r") as progress_lock:
                # If a channel was in progress, it cannot be recovered. Delete and start over.
                if progress_lock.readlines()[0][0] == "1":
                    print("Deleting already existing channel")
                    shutil.rmtree("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/")
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id), exist_ok=True)
                else:
                    #Channel is already properly indexed. Stop now.
                    print("Channel was already indexed. Stopping and returning.")
                    model_users -= 1
                    print(model_users)
                    return
        with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt", "w") as progress_lock:
            print("writing progress lock")
            # If a channel indexing is in progress and we try to search, we need to send an error to the user.
            progress_lock.write("1")
        print("Starting channel scan")
        async for message in channel.history(limit=None):
            print(".", end='', flush=True)
            if message.attachments and message.attachments != []:
                if type(channel) == discord.TextChannel:
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + str(message.id), exist_ok=True)
                    for idx, attachment in enumerate(message.attachments):
                        if attachment.content_type in ["image/jpeg", "image/png"]:
                            print("x", flush=True, end='')
                            model_loader()
                            valid = False
                            with io.BytesIO() as imagebn:
                                try:
                                    await attachment.save(fp=imagebn)
                                except:
                                    try:
                                        await attachment.save(fp=imagebn, use_cached=True)
                                    except:
                                        valid = False
                                imagebn.seek(0)
                                embeds = model.encode(Image.open(imagebn))
                            if valid: np.save("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy", embeds)
                                    
                elif type(channel) == discord.Thread:
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/threads", exist_ok=True)
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(channel.id), exist_ok=True)
                    for idx, attachment in enumerate(message.attachments):
                        if attachment.content_type in ["image/jpeg", "image/png"]:
                            print("x", flush=True, end='')
                            model_loader()
                            valid = True
                            with io.BytesIO() as imagebn:
                                try:
                                    await attachment.save(fp=imagebn)
                                except:
                                    try:
                                        await attachment.save(fp=imagebn, use_cached=True)
                                    except:
                                        valid = False
                                imagebn.seek(0)
                                embeds = model.encode(Image.open(imagebn))
                            if valid: np.save("./index/" + str(channel.guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(channel.id) + "/" + str(idx) + ".npy", embeds)
        with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt", "w") as progress_lock:
            progress_lock.write("0")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        pass #threading stuff is REALLY important to not break
    model_users -= 1
    if model_users <= 0:
        model_users = 0 # just in case
        model_loading = True
        model.to('cpu')
        model_loading = False

def call_read_channel(channel):
    #this function might not have much reason to exist
    print("running call_read_channel", channel)
    #channel = client.get_channel(channel)
    if channel is not None:
        asyncio.run_coroutine_threadsafe(coro=image_indexer(channel), loop=client.loop)

def add_guild_instance(guild):
    #SHOULD BE CALLED IN A THREAD. otherwise, event loop blocks and it never completes
    global model_users
    if not os.path.isdir("./index/" + str(guild.id)):
        os.makedirs("./index/" + str(guild.id), exist_ok=True)
    for channel in guild.channels:
        print(channel)
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
        print(guild)
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
    terms = terms.split(",")
    terms = [x.strip() for x in terms]
    model_loader()
    term_embeds = model.encode(terms)
    paths = []
    embeds = []
    for path in Path('./index/' + str(interaction.guild.id)).rglob('*.npy'):
        paths.append(path)
        embeds.append(np.load(path))
    scores = ST_util.dot_score(np.array(term_embeds), np.array(embeds))
    values, idxs = torch.topk(scores, 10)
    print(values, idxs)
    images = []
    for idx in idxs[0]:
        print(idx)
        if isinstance(idx.item(), int):
            print("appended")
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

client.run(TOKEN)
