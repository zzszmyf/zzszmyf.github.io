#!/usr/bin/env python3
"""把 zzszmyf.github.io 的 URL 推送给 IndexNow。

IndexNow 是 Bing / Yandex / Seznam / Naver / Yep 共用的「主动提交」协议：
网站所有者拿托管在自己域名下的 key 文件证明身份，然后 POST 一批 URL，
搜索引擎会在几分钟内去抓，而不是等自然爬取（这个站没有外链，自然爬取以周计）。

key 文件是 static/2b463038c3a198f501e25601eb316cc4.txt，构建后位于
https://zzszmyf.github.io/2b463038c3a198f501e25601eb316cc4.txt，line 内容必须与该 key 一致。

用法：
  # 提交某次 push 里改动过的笔记（CI 用）
  python3 .github/scripts/indexnow.py --changed <before_sha> <after_sha>

  # 提交全站（首次接入、或大改模板后手动跑）
  python3 .github/scripts/indexnow.py --all

  # 只提交指定 URL
  python3 .github/scripts/indexnow.py --urls https://zzszmyf.github.io/notes/foo/

  # 只看会提交什么，不发请求
  python3 .github/scripts/indexnow.py --all --dry-run

只提交 200 的 URL；URL 一律取 sitemap 里的百分号编码形式。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

SITE = "https://zzszmyf.github.io"
HOST = "zzszmyf.github.io"
KEY = "2b463038c3a198f501e25601eb316cc4"
KEY_LOCATION = f"{SITE}/{KEY}.txt"
LOCAL_SITEMAP = "public/sitemap.xml"

# 同一份 payload 发给这两个端点即可覆盖 Bing（前者会转发给参与 IndexNow 的所有引擎）。
ENDPOINTS = ["https://api.indexnow.org/indexnow", "https://www.bing.com/indexnow"]


def content_files_to_urls(paths: list[str]) -> list[str]:
    """把 content/ 下的文件路径换算成 Hugo 生成的小写 URL。

    Hugo 规则（本站没有 slug:/url: 覆盖，所以是纯机械换算）：
      content/notes/LLM量化精读笔记-04-量化粒度校准与离群值.md
        -> https://zzszmyf.github.io/notes/llm量化精读笔记-04-量化粒度校准与离群值/
    """
    urls = []
    for path in paths:
        if not path.startswith("content/") or not path.endswith(".md"):
            continue
        rel = path[len("content/") : -len(".md")]
        if rel.endswith("/_index") or rel == "_index":
            rel = rel[: -len("_index")].rstrip("/")
        slug = rel.lower().replace(" ", "-")
        url_path = "/" + slug.strip("/") + "/" if slug.strip("/") else "/"
        urls.append(SITE + urllib.parse.quote(url_path, safe="/"))
    return urls


def changed_files(before: str, after: str) -> list[str]:
    """列出这次 push 新增/修改（且现在仍存在）的 content 文件。"""
    # -z + core.quotePath=false：非 ASCII 文件名（本站大多数笔记都是中文名）默认会被
    # git 转义成 "\344\272\224..." 这种八进制串，直接按行读会拿到不存在的路径，静默漏提交。
    diff = subprocess.run(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=AM",
            before,
            after,
            "--",
            "content/",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    import os

    paths = [p for p in diff.split("\0") if p.strip()]
    return [p for p in paths if os.path.exists(p)]


def sitemap_urls() -> list[str]:
    """读本地构建产物，没有就抓线上 sitemap。"""
    import os

    if os.path.exists(LOCAL_SITEMAP):
        with open(LOCAL_SITEMAP, encoding="utf-8") as fh:
            xml = fh.read()
    else:
        with urllib.request.urlopen(f"{SITE}/sitemap.xml", timeout=30) as resp:
            xml = resp.read().decode("utf-8")
    return re.findall(r"<loc>(.*?)</loc>", xml)


def submit(urls: list[str], dry_run: bool = False) -> int:
    """POST 给所有端点；返回非 0 表示至少有一个端点明确报错。"""
    if not urls:
        print("没有需要提交的 URL（这次 push 没动 content/）")
        return 0
    if dry_run:
        print(f"[dry-run] 会提交 {len(urls)} 条 URL：")
        for url in urls:
            print("   ", url)
        return 0

    payload = {
        "host": HOST,
        "key": KEY,
        "keyLocation": KEY_LOCATION,
        "urlList": urls,
    }
    body = json.dumps(payload).encode("utf-8")
    print(f"向 IndexNow 提交 {len(urls)} 条 URL")

    failed = 0
    for endpoint in ENDPOINTS:
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                print(f"  {endpoint} -> HTTP {resp.status}")
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            print(f"  {endpoint} -> HTTP {err.code} {detail}")
            failed = 1
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="提交 URL 给 IndexNow")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="提交 sitemap 里的全部 URL")
    group.add_argument(
        "--changed", nargs=2, metavar=("BEFORE_SHA", "AFTER_SHA"), help="只提交这次 push 改动的笔记"
    )
    group.add_argument("--urls", nargs="+", metavar="URL", help="提交指定 URL")
    parser.add_argument("--dry-run", action="store_true", help="只打印将提交的 URL，不发请求")
    args = parser.parse_args()

    if args.all:
        urls = sitemap_urls()
    elif args.urls:
        urls = args.urls
    else:
        urls = content_files_to_urls(changed_files(*args.changed))
    return submit(urls, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
