import argparse
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import praw
import pandas as pd
from tqdm import tqdm

load_dotenv(Path.home() / ".env")

SUBREDDITS = [

    # ==========================
    # Ghost / Paranormal
    # ==========================
    "Ghoststories",
    "Paranormal",
    "Thetruthishere",
    "Humanoidencounters",
    "HighStrangeness",
    "Glitch_in_the_Matrix",
    "LetsNotMeet",
    "nosleep",
    "shortscarystories",
    "scarystories",
    "creepyencounters",
    "BackwoodsCreepy",
    "UnresolvedMysteries",
    "Missing411",

    # ==========================
    # Personal Stories
    # ==========================
    "TrueOffMyChest",
    "offmychest",
    "self",
    "CasualConversation",
    "BenignExistence",
    "Life",
    "LifeProTips",
    "TalesFromRetail",
    "TalesFromYourServer",
    "TalesFromTechSupport",

    # ==========================
    # Relationship Stories
    # ==========================
    "relationships",
    "relationship_advice",
    "dating",
    "BreakUps",
    "Marriage",
    "Love",
    "LongDistance",

    # ==========================
    # Moral Dilemmas
    # ==========================
    "AmItheAsshole",
    "AITAH",
    "amiwrong",
    "AmIOverreacting",
    "TwoHotTakes",
    "Advice",
    "NeedAdvice",

    # ==========================
    # Revenge Stories
    # ==========================
    "pettyrevenge",
    "ProRevenge",
    "NuclearRevenge",
    "MaliciousCompliance",

    # ==========================
    # Confessions
    # ==========================
    "confession",
    "confessions",
    "TrueOffMyChest",
    "offmychest",

    # ==========================
    # Workplace
    # ==========================
    "antiwork",
    "jobs",
    "work",
    "AskHR",

    # ==========================
    # Family Stories
    # ==========================
    "raisedbynarcissists",
    "insaneparents",
    "entitledparents",
    "JUSTNOMIL",
    "JUSTNOFAMILY",

    # ==========================
    # Funny / Weird
    # ==========================
    "todayilearned",
    "mildlyinteresting",
    "Wellthatsucks",
    "facepalm",
    "TIFU",

    # ==========================
    # Crime / Mystery
    # ==========================
    "RBI",
    "UnresolvedMysteries",
    "WithoutATrace",
    "TrueCrimeDiscussion",

    # ==========================
    # Survival / Adventure
    # ==========================
    "Camping",
    "Hiking",
    "Survival",
    "Backpacking",

    # ==========================
    # Ask-style Story Posts
    # ==========================
    "AskReddit",
    "AskMen",
    "AskWomen",
    "NoStupidQuestions",

    # ==========================
    # Emotional Stories
    # ==========================
    "MomForAMinute",
    "DadForAMinute",
    "KindVoice",
    "DecidingToBeBetter"
]
MIN_CHARS = 500
POST_LIMIT_PER_SORT = 1000
SLEEP_SECONDS = 0.3

def clean_text(text):
    text = re.sub(r"http\S+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def require_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

def make_reddit():
    require_env(["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"])
    return praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"), client_secret=os.getenv("REDDIT_CLIENT_SECRET"), user_agent=os.getenv("REDDIT_USER_AGENT"))

def load_paths(config_path):
    from storygen.utils import load_config_module, path_from_config, resolve_cli_path

    config = load_config_module(config_path)
    out_dir = resolve_cli_path(config, "", fallback_key="raw_data_dir", fallback="output")
    raw_data_file = resolve_cli_path(config, "", fallback_key="raw_data_file", fallback=out_dir / "ghost_stories.jsonl")
    return {
        "out_dir": out_dir,
        "jsonl": raw_data_file,
        "csv": out_dir / "ghost_stories.csv",
        "seen": out_dir / "seen_ids.txt",
    }

def load_seen(seen_file):
    return set(seen_file.read_text(encoding="utf-8").splitlines()) if seen_file.exists() else set()

def save_seen(post_id, seen_file):
    with seen_file.open("a", encoding="utf-8") as f:
        f.write(post_id + "\n")

def valid_post(post):
    if not post.is_self or post.stickied or post.over_18:
        return False
    body = post.selftext or ""
    if body in ["[removed]", "[deleted]"]:
        return False
    return len(body.strip()) >= MIN_CHARS

def post_to_record(post):
    return {
        "id": post.id,
        "subreddit": str(post.subreddit),
        "title": clean_text(post.title),
        "story": clean_text(post.selftext),
        "score": post.score,
        "upvote_ratio": post.upvote_ratio,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "permalink": f"https://www.reddit.com{post.permalink}"
    }

def iter_posts(subreddit):
    yield from subreddit.top(time_filter="all", limit=POST_LIMIT_PER_SORT)
    yield from subreddit.top(time_filter="year", limit=POST_LIMIT_PER_SORT)
    yield from subreddit.new(limit=POST_LIMIT_PER_SORT)

def scrape(paths):
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    reddit = make_reddit()
    seen = load_seen(paths["seen"])
    saved = 0

    with paths["jsonl"].open("a", encoding="utf-8") as f:
        for sub_name in SUBREDDITS:
            subreddit = reddit.subreddit(sub_name)
            for post in tqdm(iter_posts(subreddit), desc=sub_name):
                if post.id in seen or not valid_post(post):
                    continue
                record = post_to_record(post)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                save_seen(post.id, paths["seen"])
                seen.add(post.id)
                saved += 1
                time.sleep(SLEEP_SECONDS)

    print(f"Saved new posts: {saved}")

def export_csv(paths):
    if not paths["jsonl"].exists():
        print("No JSONL file found.")
        return

    rows = [json.loads(line) for line in paths["jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    df.to_csv(paths["csv"], index=False)
    print(f"CSV saved: {paths['csv']} | rows: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Reddit stories into the configured dataset location.")
    parser.add_argument("--config", default="config.py")
    args = parser.parse_args()
    paths = load_paths(args.config)
    scrape(paths)
    export_csv(paths)
