"""
generate_dataset.py
--------------------
Builds a labeled training dataset of product reviews (genuine vs fake).

IMPORTANT (read this):
This generator produces a large, diverse, *synthetic* dataset built from
patterns that peer-reviewed research (Ott et al. 2011 "Finding Deceptive
Opinion Spam"; Mukherjee et al. 2013 on Yelp fraud) has shown reliably
separate deceptive from genuine reviews:
  - Fake reviews: more superlatives/extreme sentiment, more first-person
    narrative ("I", "my"), fewer concrete/specific product details,
    shorter or oddly generic, unnatural repetition, review "bursts",
    rating/text sentiment mismatch, low reviewer history.
  - Genuine reviews: more nouns/specific details (size, part names, use
    case), mixed sentiment (pros AND cons), moderate length, natural
    imperfections, verified purchase.

This is a strong starting point and works out of the box, but for a
production system you should fine-tune/re-train on a real labeled
dataset (e.g. the Amazon Fake Review Dataset on Kaggle, or the Yelp/Ott
deceptive-opinion-spam corpus) using the same train_model.py pipeline.
Swap in real data by producing a CSV with the same columns this script
outputs: text, rating, verified_purchase, reviewer_review_count, label.
"""

import random
import csv
import numpy as np

random.seed(42)
np.random.seed(42)

PRODUCTS = [
    "wireless earbuds", "yoga mat", "coffee maker", "laptop stand", "running shoes",
    "kitchen knife set", "phone case", "bluetooth speaker", "office chair", "backpack",
    "air fryer", "electric toothbrush", "desk lamp", "water bottle", "gaming mouse",
    "winter jacket", "blender", "skincare cream", "board game", "fitness tracker",
]

# ---------- building blocks for GENUINE reviews ----------
GENUINE_POS_DETAILS = [
    "the battery lasted about {n} hours on a single charge",
    "it took roughly {n} minutes to set up out of the box",
    "the stitching on the seams looks solid after {n} weeks of daily use",
    "it fits comfortably even after wearing it for {n} hours straight",
    "the noise cancellation works well in a coffee shop but not on a plane",
    "the handle got a little loose after about {n} uses",
    "compared to the previous version I owned, this one is noticeably quieter",
    "the size ran slightly small so I'd recommend ordering one size up",
    "the manual was confusing at first but customer support replied within a day",
    "it works fine with my {n}-year-old router without any dropouts",
    "the color in person is a bit darker than the product photos",
    "one of the buttons feels a little mushy but everything else works as expected",
]
GENUINE_NEG_DETAILS = [
    "the plastic clip snapped after about {n} days",
    "shipping took longer than the estimate, about {n} extra days",
    "the instructions were missing a page which was frustrating",
    "it's a bit heavier than I expected from the listing photos",
    "the app that pairs with it crashes occasionally on Android",
    "customer service took {n} days to respond to my email",
]
GENUINE_OPENERS = [
    "I've had this for about {n} weeks now.",
    "Picked this up for {occasion}.",
    "Bought this to replace my old one which finally broke.",
    "Been using this daily for {n} days so wanted to leave an honest update.",
    "Ordered this after comparing a few similar options.",
]
OCCASIONS = ["a work trip", "my home office setup", "my daughter's birthday", "a gym routine", "camping season"]

def make_genuine_review():
    n1, n2 = random.randint(2, 30), random.randint(1, 12)
    opener = random.choice(GENUINE_OPENERS).format(n=n1, occasion=random.choice(OCCASIONS))
    n_details = random.randint(2, 4)
    details = random.sample(GENUINE_POS_DETAILS, k=min(n_details, len(GENUINE_POS_DETAILS)))
    details = [d.format(n=random.randint(1, 20)) for d in details]
    include_neg = random.random() < 0.55
    neg = ""
    if include_neg:
        neg = " " + random.choice(GENUINE_NEG_DETAILS).format(n=random.randint(1, 10)) + "."
    body = opener + " " + " ".join(d.capitalize() + "." for d in details) + neg
    closer_pool = [
        "Overall it does the job.", "Would probably buy again but not thrilled either way.",
        "Solid pick for the price.", "Not perfect but good value.",
        "Might update this review after a few more months of use.", "",
    ]
    body += " " + random.choice(closer_pool)
    rating = random.choice([3, 3, 4, 4, 4, 5, 5, 2]) if include_neg else random.choice([4, 4, 5, 5, 5])
    return body.strip(), rating

# ---------- building blocks for FAKE reviews ----------
FAKE_SUPERLATIVES = [
    "AMAZING", "the best purchase I have EVER made", "absolutely PERFECT", "life changing",
    "incredible quality", "exceeded all my expectations", "5 stars is not enough",
    "you will not regret buying this", "highly highly recommend", "beyond amazing",
]
FAKE_GENERIC = [
    "Great product, great price, great service!",
    "I love it so much, it works perfectly every single time!",
    "This is exactly what I was looking for, I am so happy I bought it!",
    "Amazing quality and fast shipping, will definitely buy again!",
    "Best product ever, I recommend it to everyone I know!",
    "Five stars all the way, this exceeded my expectations completely!",
    "I am extremely satisfied, this product is simply the best on the market!",
    "Perfect in every way, I use it every day and it never disappoints!",
]
FAKE_OPENERS = [
    "OMG I can't believe how good this is!!!",
    "As a verified buyer I have to say this is",
    "I was skeptical at first but now I'm a believer,",
    "Let me tell you,",
    "Honestly,",
]

def make_fake_review():
    style = random.random()
    if style < 0.45:
        # short, generic, superlative-heavy, low detail
        n_sent = random.randint(1, 2)
        body = " ".join(random.sample(FAKE_GENERIC, k=n_sent))
    else:
        opener = random.choice(FAKE_OPENERS)
        sup = random.sample(FAKE_SUPERLATIVES, k=random.randint(2, 3))
        body = f"{opener} {', '.join(sup)}!!! I would give it 10 stars if I could. Buy it now, you won't be disappointed!!!"
    if random.random() < 0.3:
        body += " " + random.choice(FAKE_GENERIC)
    rating = random.choice([5, 5, 5, 5, 1])  # extreme ratings, mostly 5, occasional review-bombing 1
    return body.strip(), rating

def build_dataset(n_per_class=2500):
    rows = []
    for _ in range(n_per_class):
        product = random.choice(PRODUCTS)
        text, rating = make_genuine_review()
        text = text.replace("this", f"this {product}", 1) if random.random() < 0.3 else text
        verified = 1 if random.random() < 0.85 else 0
        reviewer_count = int(np.random.gamma(shape=2.0, scale=15))  # genuine reviewers: varied history
        rows.append([text, rating, verified, reviewer_count, 0])

    for _ in range(n_per_class):
        product = random.choice(PRODUCTS)
        text, rating = make_fake_review()
        text = text.replace("this", f"this {product}", 1) if random.random() < 0.2 else text
        verified = 1 if random.random() < 0.35 else 0  # fake reviews less often verified
        reviewer_count = int(np.random.gamma(shape=1.0, scale=2))  # fake reviewers: thin history
        rows.append([text, rating, verified, reviewer_count, 1])

    random.shuffle(rows)
    return rows

if __name__ == "__main__":
    rows = build_dataset(n_per_class=2500)
    with open("reviews_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "rating", "verified_purchase", "reviewer_review_count", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to reviews_dataset.csv")
    print("label 0 = genuine, label 1 = fake")
