import typer
import json
from typing_extensions import Annotated
import httpx
import tqdm
import asyncio
from transformers import pipeline

app = typer.Typer()

# 创建一个异步 HTTP 客户端实例，设置超时为 None，表示不限制请求时间
client = httpx.AsyncClient(timeout=None)

# 定义一个异步函数 run，接受消息列表和 URL 作为参数
async def run(messages: list, url: str):
    payload = {"model":"tgi", "messages": messages}
    response = await client.post(url, json=payload)
    content = response.json()
    message = content["choices"][0]["message"]
    message.pop("name", None)
    messages.append(message)
    return messages

def fix_source(source):
    if source and source[0]["from"] == "gpt":
        # Skip if GPT is first to talk
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source

async def recreate_conversation(conversation, sem, url):
    async with sem:
        messages = []
        try:
            for message in conversation[::2]:
                assert message["role"] == "user"
                messages.append(message)
                messages = await run(messages, url)
        except Exception as e:
            print(e)
            pass
        return messages


@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    url: Annotated[str, typer.Option("--url")] = "http://localhost:8080/v1/chat/completions",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 64
):
    """
    主函数：处理输入文件中的对话并生成新的对话
    """
    async def _main():
        # 确保使用新的事件循环
        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(concurrency)
        
        # 读取输入文件
        with open(input_filename, "r") as f:
            input_data = json.loads(f.read())
        conversations = [fix_source(source["conversations"]) for source in input_data]

        futures = []
        for conversation in conversations:
            future = recreate_conversation(conversation, sem, url)
            futures.append(future)

        # 使用 asyncio.gather 替代 tqdm.asyncio.tqdm.gather
        recreated_conversations = []
        success_count = 0
        total = len(futures)

        for future in tqdm.tqdm(asyncio.as_completed(futures), total=total):
            try:
                result = await future
                recreated_conversations.append(result)
                success_count += 1
            except Exception as e:
                print(f"处理失败: {str(e)}")

        print(f"\n总共: {total}, 成功: {success_count}, 失败: {total - success_count}")

        with open(output_filename, "w") as f:
            json.dump(recreated_conversations, f, indent=4)

    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_main())
    finally:
        loop.close()


if __name__ == "__main__":
    app()
