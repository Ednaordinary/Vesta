import nextcord as discord
import sys
import time
import io
import os
import gc
import vram
import asyncio
from typing import Optional
from dotenv import load_dotenv
import threading
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as ST_util
from PIL import Image, ImageFile
import numpy as np
import shutil
from pathlib import Path
import torch
import requests
import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.AutoShardedClient(intents=intents)
#model = SentenceTransformer('clip-ViT-L-14').to("cpu")
#model = SentenceTransformer('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k').to("cpu")
model = None
model_users = 0
model_queue = []
searchers = 0
download_queue = []
messages = {}
image_attachments = {}

class term_colors:
    LOAD = '\033[96m'  # On load or unload state
    DOWNLOAD = '\033[94m'  # On submission to the attachment downloaded
    ENCODER = '\033[92m'  # Encoder info
    END = '\033[0m'  #End coloring


def download_thread(url, path):
    global model_queue
    try:
        content = requests.get(url).content
        img = Image.open(io.BytesIO(content))
        model_queue.append(tuple((img, path)))
    except:
        pass


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
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_encoder())


async def async_encoder():
    global model
    global model_queue
    global searchers
    model = None
    gc.collect()
    while True:
        while model_queue == []:
            if model and searchers <= 0:
                searchers = 0
                model = None
                gc.collect()
                vram.deallocate("Vesta")
            time.sleep(0.02)
        searchers += 1  # the searcher thread will also try to handle model migration. stop it
        if not model:
            vram.allocate("Vesta")
            async for i in vram.wait_for_allocation("Vesta"):
                pass
            model = SentenceTransformer('laion/bigG', local_files_only=True, cache_folder="./",
                                        model_kwargs=dict(attn_implementation="flash_attention_2"))
            print(term_colors.LOAD + " loading encoder on cpu " + term_colors.END, end='')
        now = [x for x in model_queue]
        model_queue = []
        paths = []
        contents = []
        print(term_colors.ENCODER + "e:" + str(len(now)) + term_colors.END, end='')
        for url, path in now:
            paths.append(path)
            contents.append(url)
        if len(contents) != 0:
            try:
                embeds = model.encode(contents)
                for idx, embed in enumerate(embeds):
                    np.save(paths[idx], embed)
                    print(term_colors.ENCODER + "^" + term_colors.END, flush=True, end='')
            except:
                pass
        searchers -= 1
        if model_queue == []:
            print("\nencoder caught up!")


async def image_indexer(channel):
    global model_users
    global download_queue
    try:
        last_time = None
        if os.path.exists("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt"):
            with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt",
                      "r") as progress_lock:
                try:
                    time_stamp = progress_lock.readlines()[0]
                except:
                    time_stamp = "0"
                if time_stamp in ["0", "1"]:
                    last_time = None
                else:
                    last_time = datetime.datetime.fromtimestamp(float(time_stamp))
        os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id), exist_ok=True)
        async for message in channel.history(limit=None, after=last_time):
            print(".", end='', flush=True)
            if message.attachments and message.attachments != [] and not message.author.bot:
                if type(channel) == discord.TextChannel:
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + str(message.id),
                                exist_ok=True)
                    for idx, attachment in enumerate(message.attachments):
                        if attachment.content_type in ["image/jpeg", "image/png"]:
                            print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                            download_queue.append(tuple((attachment, "./index/" + str(channel.guild.id) + "/" + str(
                                channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))
                elif type(channel) == discord.Thread:
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id), exist_ok=True)
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/threads", exist_ok=True)
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(
                        channel.id), exist_ok=True)
                    for idx, attachment in enumerate(message.attachments):
                        if attachment.content_type in ["image/jpeg", "image/png"]:
                            print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                            download_queue.append(tuple((attachment, "./index/" + str(channel.guild.id) + "/" + str(
                                channel.parent.id) + "/threads/" + str(channel.id) + "/" + str(idx) + ".npy")))
        with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt",
                  "w") as progress_lock:
            progress_lock.write(str(time.time()))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        pass  #threading stuff is REALLY important to not break
    model_users -= 1


def call_read_channel(channel):
    #this function might not have much reason to exist
    if channel is not None:
        asyncio.run_coroutine_threadsafe(coro=image_indexer(channel), loop=client.loop)


def add_guild_instance(guild):
    #SHOULD BE CALLED IN A THREAD. otherwise, event loop blocks and it never completes
    global model_users
    if not os.path.isdir("./index/" + str(guild.id)):
        os.makedirs("./index/" + str(guild.id), exist_ok=True)
    for channel in guild.channels:
        if model_users > 5:
            while model_users > 5:
                time.sleep(0.1)
        model_users += 1
        if type(channel) == discord.TextChannel:
            os.makedirs("./index/" + str(guild.id) + "/" + str(channel.id), exist_ok=True)
            call_read_channel(channel)
        elif type(channel) == discord.ForumChannel:
            os.makedirs("./index/" + str(guild.id) + "/" + str(channel.id), exist_ok=True)
            for thread in channel.threads:
                os.makedirs("./index/" + str(guild.id) + "/" + str(thread.parent.id) + "/threads/" + str(thread.id),
                            exist_ok=True)
                call_read_channel(thread)
        elif type(channel) == discord.Thread:
            os.makedirs("./index/" + str(guild.id) + "/" + str(channel.parent.id), exist_ok=True)
            os.makedirs("./index/" + str(guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(channel.id),
                        exist_ok=True)
            call_read_channel(channel)
        else:
            model_users -= 1
    while model_users > 0:
        time.sleep(0.01)


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


discord_attacher = {}


def image_search_download(interaction, attachment, idx):
    #multi thread this because downloading takes a long time
    global discord_attacher
    file_end = ""
    if attachment.content_type == "image/jpeg":
        file_end = ".jpg"
    elif attachment.content_type == "image/png":
        file_end = ".png"
    new_image = Image.open(io.BytesIO(requests.get(attachment.url).content)).convert(mode='RGB')
    new_image.thumbnail((768, 768))
    imageio = io.BytesIO()
    new_image.save(imageio, format='JPEG')
    imageio.seek(0)
    discord_attacher[interaction][idx] = discord.File(fp=imageio,
                                                      filename=str(idx) + file_end)


def image_search_loader(interaction, path):
    global discord_attacher
    discord_attacher[interaction].append((path, np.load(path)))


def image_search(interaction, term):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_image_search(interaction, term))

def message_fetch(interaction, channel, message, number, index):
    global messages
    global image_attachments
    channel = client.get_channel(channel)
    message = asyncio.run_coroutine_threadsafe(coro=channel.fetch_message(message),
                                               loop=client.loop).result()
    image_attachments[interaction][index] = message.attachments[number]
    messages[interaction][index] = message

async def async_image_search(interaction, term):
    #Changed to a different thread because otherwise it blocks and concurrent requests aren't handled
    global searchers
    global model
    global discord_attacher
    global messages
    global image_attachments
    term = term.strip()
    searchers += 1
    vram.allocate("Vesta")
    async for i in vram.wait_for_allocation("Vesta"):
        asyncio.run_coroutine_threadsafe(
            coro=interaction.edit_original_message(content="Waiting for " + i + " before loading model."),
            loop=client.loop)
    start_time = time.time()
    print("starting compute")
    model = SentenceTransformer('laion/bigG', local_files_only=True, cache_folder="./",
                                model_kwargs=dict(attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu"))
    asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Searching... Model Loaded"), loop=client.loop)
    term_embeds = model.encode(term)
    searchers -= 1
    discord_attacher[interaction] = []
    threads = []
    for path in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("index/" + str(interaction.guild.id)))
                 for f in fn]:
        if "npy" in str(path):
            threads.append(threading.Thread(target=image_search_loader, args=[interaction, str(path)]))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    paths = []
    embeds = []
    for emb_idx in discord_attacher[interaction]:
        paths.append(emb_idx[0])
        embeds.append(emb_idx[1])
    del discord_attacher[interaction]
    print("compute time up to similarity:", time.time() - start_time)
    asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Searching... Finding similarities"),
                                     loop=client.loop)
    scores = ST_util.dot_score(torch.tensor(np.array(term_embeds), device="cpu"),
                               torch.tensor(np.array(embeds), device="cpu"))
    values, idxs = torch.topk(scores, 10)
    print("compute time:", time.time() - start_time)
    images = []
    for idx in idxs[0]:
        if isinstance(idx.item(), int):
            images.append(paths[int(idx)])
    image_attachments[interaction] = [None]*len(images)
    messages[interaction] = [None]*len(images)
    message_fetcher_threads = []
    index = 0
    for path in images:
        path = str(path)
        try:
            if "threads" in path:
                message_fetcher_threads.append(threading.Thread(target=message_fetch, args=[interaction, int(path.split("/")[4]), int(path.split("/")[5]), int(path.split("/")[6].split(".")[0]), index]))
                index += 1
            else:
                message_fetcher_threads.append(threading.Thread(target=message_fetch, args=[interaction, int(path.split("/")[2]), int(path.split("/")[3]), int(path.split("/")[4].split(".")[0]), index]))
                index += 1
        except Exception as e:
            print("failed to find!")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(repr(e))
            pass
    for message_fetcher in message_fetcher_threads:
        message_fetcher.start()
    for message_fetcher in message_fetcher_threads:
        message_fetcher.join()
    asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Results:" + "".join(
        [("\n" + str(ind + 1) + " - " + str(x)) for ind, x in enumerate([x.jump_url for x in messages[interaction] if x])])), loop=client.loop)
    download_threads = []
    discord_attacher[interaction] = [0] * len(image_attachments[interaction])
    for idx, attachment in enumerate(image_attachments[interaction]):
        download_threads.append(threading.Thread(target=image_search_download, args=[interaction, attachment, idx]))
    for thread in download_threads:
        thread.start()
    for thread in download_threads:
        thread.join()
    results = []
    for result in discord_attacher[interaction]:
        if type(result) == discord.File:
            results.append(result)
    asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Results:" + "".join(
        [("\n" + str(ind + 1) + " - " + str(x)) for ind, x in enumerate([x.jump_url for x in messages[interaction] if x])]),
                                                                            files=results), loop=client.loop)
    del discord_attacher[interaction], message_fetcher_threads, messages[interaction], image_attachments[interaction]
    gc.collect()


@client.event
async def on_message(message):
    if message.type != discord.MessageType.default and message.type != discord.MessageType.reply:
        return
    global model_users
    if message.attachments and message.attachments != [] and not message.author.bot:
        if type(message.channel) == discord.TextChannel:
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/" + str(message.id),
                        exist_ok=True)
            for idx, attachment in enumerate(message.attachments):
                if attachment.content_type in ["image/jpeg", "image/png"]:
                    print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                    download_queue.append(tuple((attachment, "./index/" + str(message.guild.id) + "/" + str(
                        message.channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))
        elif type(message.channel) == discord.Thread:
            if model_users == 0:  # update time only if indexer threads arent running (what if we Ctrl+c out after an on_message has been written but still indexing and all those messages dont get indexed?)
                with open("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/" + "inprogress.txt",
                          "w") as progress_lock:
                    progress_lock.write(str(time.time()))
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.id), exist_ok=True)
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/threads", exist_ok=True)
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.parent.id) + "/threads/" + str(
                message.channel.id), exist_ok=True)
            for idx, attachment in enumerate(message.attachments):
                if attachment.content_type in ["image/jpeg", "image/png"]:
                    print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                    download_queue.append(tuple((attachment, "./index/" + str(message.guild.id) + "/" + str(
                        message.channel.parent.id) + "/threads/" + str(message.channel.id) + "/" + str(idx) + ".npy")))


@client.event
async def on_raw_message_delete(payload):
    channel_id = payload.channel_id
    guild_id = payload.guild_id
    message_id = payload.message_id
    channel = client.get_channel(channel_id)
    if type(channel) == discord.TextChannel:
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel_id) + "/" + str(message_id) + "/"):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel_id) + "/" + str(message_id) + "/")
    elif type(channel) == discord.Thread:
        channel = client.get_channel(channel_id)
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id)):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id))


@client.event
async def on_raw_message_edit(payload):
    global model_users
    channel_id = payload.channel_id
    guild_id = payload.guild_id
    message_id = payload.message_id
    channel = client.get_channel(channel_id)
    if type(channel) == discord.TextChannel:
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel_id) + "/" + str(message_id) + "/"):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel_id) + "/" + str(message_id) + "/")
        message = await channel.fetch_message(message_id)
        if message.type != discord.MessageType.default and message.type != discord.MessageType.reply:
            return
        if message.attachments and message.attachments != [] and not message.author.bot:
            if model_users == 0:  # update time only if indexer threads arent running (what if we Ctrl+c out after an on_message has been written but still indexing and all those messages dont get indexed?)
                with open("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/" + "inprogress.txt",
                          "w") as progress_lock:
                    progress_lock.write(str(time.time()))
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/" + str(message.id),
                        exist_ok=True)
            for idx, attachment in enumerate(message.attachments):
                if attachment.content_type in ["image/jpeg", "image/png"]:
                    print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                    download_queue.append(tuple((attachment, "./index/" + str(message.guild.id) + "/" + str(
                        message.channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))
    elif type(channel) == discord.Thread:
        channel = client.get_channel(channel_id)
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id)):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id))
        message = await channel.fetch_message(message_id)
        if message.type != discord.MessageType.default and message.type != discord.MessageType.reply:
            return
        if message.attachments and message.attachments != [] and message.author.id != client.user.id:
            if model_users == 0:  # update time only if indexer threads arent running (what if we Ctrl+c out after an on_message has been written but still indexing and all those messages dont get indexed?)
                with open("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/" + "inprogress.txt",
                          "w") as progress_lock:
                    progress_lock.write(str(time.time()))
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.id), exist_ok=True)
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/threads", exist_ok=True)
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.parent.id) + "/threads/" + str(
                message.channel.id), exist_ok=True)
            for idx, attachment in enumerate(message.attachments):
                if attachment.content_type in ["image/jpeg", "image/png"]:
                    print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                    download_queue.append(tuple((attachment, "./index/" + str(message.guild.id) + "/" + str(
                        message.channel.parent.id) + "/threads/" + str(message.channel.id) + "/" + str(idx) + ".npy")))


@client.slash_command(description="Search for an image with the description.")
async def find_image(
        interaction: discord.Interaction,
        term: str = discord.SlashOption(
            name="term",
            required=True,
            description="Term to search for.",
        ),
):
    await interaction.response.send_message("Searching...")
    threading.Thread(target=image_search, args=[interaction, term]).start()


threading.Thread(target=downloader).start()
threading.Thread(target=encoder).start()
client.run(TOKEN)
