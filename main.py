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
import numpy.lib.format
import struct
import shutil
from pathlib import Path
from queue import Queue, Empty
import torch
import requests
import datetime
import ctypes
import psutil

libc = ctypes.CDLL("libc.so.6") # Needed for memory management

ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.AutoShardedClient(intents=intents)
model = None
model_users = 0
model_queue = Queue()
searchers = 0
download_queue = Queue()
messages = {}
image_attachments = {}
discord_attacher = {}

class term_colors:
    LOAD = '\033[96m'  # On load or unload state
    DOWNLOAD = '\033[94m'  # On submission to the attachment downloaded
    ENCODER = '\033[92m'  # Encoder info
    END = '\033[0m'  #End coloring

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    libc.malloc_trim(0)

def download_thread(url, path):
    try:
        content = requests.get(url).content
        img = Image.open(io.BytesIO(content))
        model_queue.put(tuple((img, path)))
        del content, img
    except:
        pass


def downloader():
    while True:
        now = download_queue.get()
        url, path = now
        threading.Thread(target=download_thread, args=[url, path]).start()
        del now
        gc.collect()

def np_save(file, array):
    magic_string=b"\x93NUMPY\x01\x00v\x00"
    header=bytes(("{'descr': '"+array.dtype.descr[0][1]+"', 'fortran_order': False, 'shape': "+str(array.shape)+", }").ljust(127-len(magic_string))+"\n",'utf-8')
    if type(file) == str:
        file=open(file,"wb")
    file.write(magic_string)
    file.write(header)
    file.write(array.data)

def encoder():
    global model
    global model_queue
    global searchers
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    while True:
        while True:
            try:
                now = [model_queue.get(timeout=0.5)]
            except Empty as e:
                pass
            else:
                break
            if model and searchers <= 0:
                searchers = 0
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                vram.deallocate("Vesta")
        searchers += 1  # the searcher thread will also try to handle model migration. stop it
        if not model:
            vram.allocate("Vesta")
            for i in vram.wait_for_allocation("Vesta"):
                pass
            model = SentenceTransformer('laion/DCXL', local_files_only=True, cache_folder="./",
                                        model_kwargs=dict(attn_implementation="flash_attention_2"))
            print(term_colors.LOAD + " loading encoder " + term_colors.END, end='')
        while True:
            try:
                now.append(model_queue.get(block=False))
            except Empty as e:
                break
        for i in now[512:]:
            model_queue.put(i)
        now = now[:512]
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
                    np_save(paths[idx], embed)
                    print(term_colors.ENCODER + "^" + term_colors.END, flush=True, end='')
                del embeds
            except:
                pass
        searchers -= 1
        if model_queue.qsize() == 0:
            del now, paths, contents
            print("\nencoder caught up!")
            flush()
            print(f'Current memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.3f}GiB')


async def image_indexer(channel):
    global model_users
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
                            download_queue.put(tuple((attachment, "./index/" + str(channel.guild.id) + "/" + str(
                                channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))
                elif type(channel) == discord.Thread:
                    os.makedirs("./index/" + str(channel.guild.id) + "/" + str(channel.parent.id) + "/threads/" + str(channel.id) + "/" + str(message.id), exist_ok=True)
                    for idx, attachment in enumerate(message.attachments):
                        if attachment.content_type in ["image/jpeg", "image/png"]:
                            print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                            download_queue.put(tuple((attachment, "./index/" + str(channel.guild.id) + "/" + str(
                                channel.parent.id) + "/threads/" + str(channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))
        with open("./index/" + str(channel.guild.id) + "/" + str(channel.id) + "/" + "inprogress.txt",
                  "w") as progress_lock:
            progress_lock.write(str(time.time()))
    except Exception as e:
        #exc_type, exc_obj, exc_tb = sys.exc_info()
        #fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #print(exc_type, fname, exc_tb.tb_lineno)
        #print(repr(e))
        pass  #threading stuff is REALLY important to not break
    flush()
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
    print(f'Current memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.3f}GiB')
    os.makedirs("./index", exist_ok=True)
    for guild in client.guilds:
        threading.Thread(target=add_guild_instance, args=[guild]).start()

def np_load(file):
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(',)', ')').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

def image_search_loader(interaction, path):
    global discord_attacher
    discord_attacher[interaction].append((path, np_load(path)))


def image_search(interaction, term):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_image_search(interaction, term))

def message_fetch(interaction, channel, message, number, index):
    global messages
    global image_attachments
    channel = client.get_channel(channel)
    message = asyncio.run_coroutine_threadsafe(coro=channel.fetch_message(message),
                                               loop=client.loop).result()
    new_image = Image.open(io.BytesIO(requests.get(message.attachments[number].url).content)).convert(mode='RGB')
    new_image.thumbnail((768, 768))
    imageio = io.BytesIO()
    new_image.save(imageio, format='JPEG')
    imageio.seek(0)
    image_attachments[interaction][index] = discord.File(fp=imageio,
                                                      filename=str(index) + ".jpg")
    messages[interaction][index] = message

async def async_image_search(interaction, term):
    #Changed to a different thread because otherwise it blocks and concurrent requests aren't handled
    global searchers
    global model
    global discord_attacher
    global messages
    global image_attachments
    if isinstance(term, str):
        term = term.strip()
    searchers += 1
    vram.allocate("Vesta")
    for i in vram.wait_for_allocation("Vesta"):
        asyncio.run_coroutine_threadsafe(
            coro=interaction.edit_original_message(content="Waiting for " + i + " before loading model."),
            loop=client.loop)
    start_time = time.time()
    print("starting compute")
    if model == None:
        model = SentenceTransformer('laion/DCXL', local_files_only=True, cache_folder="./", # , 
                                model_kwargs=dict(torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True, attn_implementation="flash_attention_2"))
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
    #asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Searching... Finding similarities"),
    #                                 loop=client.loop)
    scores = ST_util.cos_sim(torch.tensor(np.array(term_embeds), device="cuda"),
                               torch.tensor(np.array(embeds), device="cuda"))
    print("compute time after similarity:", time.time() - start_time)
    #print(scores.shape)
    #deduplication
    scores = scores.cpu()
    scores = np.array([(i, x) for i, x in enumerate(scores[0])])
    print("original:", scores.shape)
    _, unique = np.unique(scores[:,1].round(decimals=6), return_index=True)
    scores = torch.Tensor(scores[unique])
    print("dedupe:", scores.shape)
    print(scores)
    #minp + topk sampling
    if len(scores) > 0:
        max_value = torch.max(scores[:,1])
        scores = [(i, x) for i, x in scores if x > (float(max_value) * 0.8)]
        scores_torch = torch.tensor([x for i, x in scores])
        values, idxs = torch.topk(scores_torch, min(10, int(scores_torch.shape[0])))
        idxs = [int(scores[x][0]) for x in idxs]
    else:
        values, idxs = [], []
    print("compute time:", time.time() - start_time)
    images = []
    for idx in idxs:
        if isinstance(idx, int):
            images.append(paths[int(idx)])
    image_attachments[interaction] = [None]*len(images)
    messages[interaction] = [None]*len(images)
    message_fetcher_threads = []
    message_links = []
    index = 0
    for path in images:
        path = str(path)
        try:
            if "threads" in path:
                message_fetcher_threads.append(threading.Thread(target=message_fetch, args=[interaction, int(path.split("/")[4]), int(path.split("/")[5]), int(path.split("/")[6].split(".")[0]), index]))
                message_links.append("https://discord.com/channels/" + str(interaction.guild.id) + "/" + str(int(path.split("/"))[4]) + "/" + str(int(path.split("/")[5])))
                index += 1
            else:
                message_fetcher_threads.append(threading.Thread(target=message_fetch, args=[interaction, int(path.split("/")[2]), int(path.split("/")[3]), int(path.split("/")[4].split(".")[0]), index]))
                message_links.append("https://discord.com/channels/" + str(interaction.guild.id) + "/" + str(int(path.split("/")[2])) + "/" + str(int(path.split("/")[3])))
                index += 1
        except Exception as e:
            print("failed to find!")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(repr(e))
            pass
    print(message_links)
    asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Results:" + "".join(
        [("\n" + str(ind + 1) + " - " + str(x)) for ind, x in enumerate(message_links)])), loop=client.loop)
    for message_fetcher in message_fetcher_threads:
        message_fetcher.start()
    for message_fetcher in message_fetcher_threads:
        message_fetcher.join()
    download_threads = []
    image_attachments[interaction] = [x for x in image_attachments[interaction] if x != None]
    #discord_attacher[interaction] = [0] * len(image_attachments[interaction])
    #for idx, attachment in enumerate(image_attachments[interaction]):
    #    download_threads.append(threading.Thread(target=image_search_download, args=[interaction, attachment, idx]))
    #for thread in download_threads:
    #    thread.start()
    #for thread in download_threads:
    #    thread.join()
    results = [x for x in image_attachments[interaction] if type(x) == discord.File]
    asyncio.run_coroutine_threadsafe(coro=interaction.edit_original_message(content="Results:" + "".join(
        [("\n" + str(ind + 1) + " - " + str(x)) for ind, x in enumerate([x.jump_url for x in messages[interaction] if x])]),
                                                                            files=results), loop=client.loop)
    del message_fetcher_threads, messages[interaction], image_attachments[interaction]
    flush()
    print(f'Current memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.3f}GiB')

def image_index(message):
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
                    download_queue.put(tuple((attachment, "./index/" + str(message.guild.id) + "/" + str(
                        message.channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))
        elif type(message.channel) == discord.Thread:
            if model_users == 0:  # update time only if indexer threads arent running (what if we Ctrl+c out after an on_message has been written but still indexing and all those messages dont get indexed?)
                with open("./index/" + str(message.guild.id) + "/" + str(message.channel.id) + "/" + "inprogress.txt",
                          "w") as progress_lock:
                    progress_lock.write(str(time.time()))
            os.makedirs("./index/" + str(message.guild.id) + "/" + str(message.channel.parent.id) + "/threads/" + str(message.channel.id) + "/" + str(message.id), exist_ok=True)
            for idx, attachment in enumerate(message.attachments):
                if attachment.content_type in ["image/jpeg", "image/png"]:
                    print(term_colors.DOWNLOAD + "x" + term_colors.END, flush=True, end='')
                    download_queue.put(tuple((attachment, "./index/" + str(message.guild.id) + "/" + str(
                        message.channel.parent.id) + "/threads/" + str(message.channel.id) + "/" + str(message.id) + "/" + str(idx) + ".npy")))

@client.event
async def on_message(message):
    image_index(message)

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
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id) + "/" + str(message_id)):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id) + "/" + str(message_id))


@client.event
async def on_raw_message_edit(payload):
    channel_id = payload.channel_id
    guild_id = payload.guild_id
    message_id = payload.message_id
    channel = client.get_channel(channel_id)
    if type(channel) == discord.TextChannel:
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel_id) + "/" + str(message_id) + "/"):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel_id) + "/" + str(message_id) + "/")
    elif type(channel) == discord.Thread:
        channel = client.get_channel(channel_id)
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id) + "/" + str(message_id)):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id) + "/" + str(message_id))
    try:
        message = await client.fetch_message(message_id)
    except:
        pass
    else:
        image_index(message)

@client.event
async def on_member_update(before, after):
    if after.id == client.user.id:
        if before.roles != after.roles:
            threading.Thread(target=add_guild_instance, args=[guild]).start()

@client.event
async def on_guild_channel_create(channel):
    call_read_channel(channel)

@client.event
async def on_guild_channel_delete(channel):
    call_read_channel(channel)
    if type(channel) == discord.TextChannel:
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel_id) + "/"):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel_id) + "/")
    elif type(channel) == discord.Thread:
        channel = client.get_channel(channel_id)
        if os.path.isdir("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id)):
            shutil.rmtree("./index/" + str(guild_id) + "/" + str(channel.parent.id) + "/threads/" + str(channel_id))

@client.slash_command()
async def find(interaction: discord.Interaction):
    pass

@find.subcommand(name="from")
async def find_from(interaction: discord.Interaction):
    pass

@find_from.subcommand(description="Search for an image based on a description.")
async def text(
        interaction: discord.Interaction,
        term: str = discord.SlashOption(
            name="term",
            required=True,
            description="Term to search for.",
        ),
):
    await interaction.response.send_message("Searching...")
    threading.Thread(target=image_search, args=[interaction, term]).start()

@find_from.subcommand(description="Search for an image based on another image.")
async def image(
        interaction: discord.Interaction,
        term: discord.Attachment
):
    if not term.content_type.lower() in ["image/png", "image/jpg", "image/jpeg"]:
        await interaction.response.send_message("Term should be a png or jpg")
        return
    try:
        termbytes = await term.read()
        term = Image.open(io.BytesIO(termbytes))
    except Exception as e:
        print(repr(e))
        await interaction.response.send_message("Something went wrong while opening the image.")
        return
    await interaction.response.send_message("Searching...")
    threading.Thread(target=image_search, args=[interaction, term]).start()
    del interaction, term


threading.Thread(target=downloader).start()
threading.Thread(target=encoder).start()
client.run(TOKEN)
