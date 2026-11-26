"""Generate the hand-written review dataset for the Roman Urdu Sentiment
Analyzer. Every sentence below was written for this project — no external
dataset is used. Run: python scripts/generate_dataset.py

Output: data/reviews.csv with columns: text, sentiment, topic, language
(sentiment in positive/negative/neutral, language in roman_urdu/english/mixed).
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rusent.preprocessing import FOLLOWUP_CLAUSES as FOLLOWUPS  # noqa: E402

random.seed(42)

POSITIVE = [
    # food
    ("khana bohat acha tha, har cheez fresh thi", "food", "roman_urdu"),
    ("pizza bilkul fresh tha aur cheese bhi zabardast thi", "food", "roman_urdu"),
    ("biryani ka taste lajawab tha, masalay bilkul perfect thay", "food", "roman_urdu"),
    ("malai boti pizza bohat tasty tha, dobara zaroor order karunga", "food", "roman_urdu"),
    ("khana garam aur fresh mila, maza aa gaya", "food", "roman_urdu"),
    ("burger bohat juicy tha aur fries bhi crispy thi", "food", "roman_urdu"),
    ("shawarma ka taste bohat acha tha, sauce bhi perfect thi", "food", "roman_urdu"),
    ("zinger burger bohat acha tha, bohat maza aaya", "food", "roman_urdu"),
    ("khana waqt par mila aur taste bhi kamal ka tha", "food", "roman_urdu"),
    ("dessert bohat sweet aur fresh tha, bohat pasand aaya", "food", "roman_urdu"),
    ("The food was amazing, everything tasted fresh and delicious", "food", "english"),
    ("Best pizza in town, the crust was perfect and the cheese was great", "food", "english"),
    ("The biryani was full of flavour and the portions were generous", "food", "english"),
    ("Really enjoyed the burger, juicy and cooked just right", "food", "english"),
    ("Fresh ingredients, great taste, I will definitely come back", "food", "english"),
    ("khana bohat tasty tha and the service was excellent too", "food", "mixed"),
    ("pizza was amazing, cheese bhi bohat fresh thi", "food", "mixed"),
    ("biryani ka taste was outstanding, highly recommended", "food", "mixed"),
    ("burger was so good, fries bhi crispy thi", "food", "mixed"),
    ("food was fresh aur garam bhi tha, maza aa gaya", "food", "mixed"),
    # shopping
    ("kapray ki quality bohat achi hai, stitching bhi perfect hai", "shopping", "roman_urdu"),
    ("shoes bohat comfortable hain, size bhi bilkul theek hai", "shopping", "roman_urdu"),
    ("yeh kurta bohat acha hai, colour bhi bilkul waisa hi hai jaisa photo mein tha", "shopping", "roman_urdu"),
    ("watch ki quality bohat achi hai, bohat khubsurat hai", "shopping", "roman_urdu"),
    ("order bilkul theek mila, packing bhi bohat achi thi", "shopping", "roman_urdu"),
    ("price ke hisaab se quality bohat achi hai, paisa wasool hai", "shopping", "roman_urdu"),
    ("The quality of this shirt is excellent, exactly as shown in the pictures", "shopping", "english"),
    ("Beautiful watch, looks premium and works perfectly", "shopping", "english"),
    ("Great value for money, the fabric feels really good", "shopping", "english"),
    ("Shoes are very comfortable and the size is perfect", "shopping", "english"),
    ("The fabric is amazing, colour bhi bilkul same hai", "shopping", "mixed"),
    ("quality is excellent, price bhi reasonable hai", "shopping", "mixed"),
    ("product bohat acha hai, I am very happy with it", "shopping", "mixed"),
    ("packing was great aur product bhi bilkul perfect tha", "shopping", "mixed"),
    # delivery / service
    ("delivery bohat fast thi, order 30 minute mein aa gaya", "delivery", "roman_urdu"),
    ("rider bohat polite tha aur order bilkul theek mila", "delivery", "roman_urdu"),
    ("customer service bohat achi hai, meri problem foran solve kar di", "delivery", "roman_urdu"),
    ("staff bohat friendly tha, bohat acha experience raha", "delivery", "roman_urdu"),
    ("booking process bohat easy tha, sab kuch smooth raha", "delivery", "roman_urdu"),
    ("Delivery was super fast, my order arrived in 25 minutes", "delivery", "english"),
    ("The rider was polite and the food was still hot when it arrived", "delivery", "english"),
    ("Excellent customer service, they solved my issue within minutes", "delivery", "english"),
    ("delivery bohat fast thi and the rider was very polite", "delivery", "mixed"),
    ("customer service was excellent, problem foran solve ho gayi", "delivery", "mixed"),
    ("staff was very friendly, bohat acha experience raha", "delivery", "mixed"),
    # mobile / app
    ("app bohat easy hai, order karna bohat asaan hai", "mobile", "roman_urdu"),
    ("yeh app bohat fast hai, koi lag nahi hai", "mobile", "roman_urdu"),
    ("app ka design bohat acha hai, sab kuch clear hai", "mobile", "roman_urdu"),
    ("new update bohat acha hai, sab kuch smooth chal raha hai", "mobile", "roman_urdu"),
    ("This app is really easy to use, everything is clear and fast", "mobile", "english"),
    ("The app works smoothly, no crashes at all", "mobile", "english"),
    ("Great app, the new update made everything even better", "mobile", "english"),
    ("app is very smooth, koi problem nahi hai", "mobile", "mixed"),
    ("interface bohat acha hai, everything is easy to find", "mobile", "mixed"),
    ("app bohat fast hai and the design is beautiful", "mobile", "mixed"),
]

NEGATIVE = [
    # food
    ("khana bilkul thanda mila, taste bhi bohat kharab tha", "food", "roman_urdu"),
    ("pizza mein cheese bohat kam thi, bilkul maza nahi aaya", "food", "roman_urdu"),
    ("biryani mein namak bohat zyada tha, khana mushkil ho gaya", "food", "roman_urdu"),
    ("burger bilkul dry tha, bilkul fresh nahi tha", "food", "roman_urdu"),
    ("khana 2 ghante late mila aur thanda bhi tha", "food", "roman_urdu"),
    ("fries bilkul soggy thi, bilkul crispy nahi thi", "food", "roman_urdu"),
    ("shawarma mein chicken bohat kam tha, sirf salad tha", "food", "roman_urdu"),
    ("khana bilkul bekaar tha, paisa zaya ho gaya", "food", "roman_urdu"),
    ("The food was cold and tasteless, a complete waste of money", "food", "english"),
    ("Worst pizza ever, the crust was burnt and there was hardly any cheese", "food", "english"),
    ("The biryani was too salty and the rice was undercooked", "food", "english"),
    ("Burger was dry and stale, I could not even finish it", "food", "english"),
    ("khana bilkul thanda tha and the taste was terrible", "food", "mixed"),
    ("pizza was so bad, cheese bhi bohat kam thi", "food", "mixed"),
    ("burger was dry aur bilkul fresh nahi tha", "food", "mixed"),
    ("taste was horrible, paisa bilkul zaya ho gaya", "food", "mixed"),
    # shopping
    ("kapray ki quality bohat kharab hai, ek wash mein colour fade ho gaya", "shopping", "roman_urdu"),
    ("shoes ka size galat bheja gaya, bilkul fit nahi aaye", "shopping", "roman_urdu"),
    ("product photo jaisa bilkul nahi hai, bohat farq hai", "shopping", "roman_urdu"),
    ("watch do din mein kharab ho gayi, bilkul bekaar hai", "shopping", "roman_urdu"),
    ("stitching bohat kharab hai, pehli dafa pehenne par phat gaya", "shopping", "roman_urdu"),
    ("order mein galat item bhej diya, bohat gussa aaya", "shopping", "roman_urdu"),
    ("The fabric quality is terrible, it faded after one wash", "shopping", "english"),
    ("Wrong size delivered and the return process is a nightmare", "shopping", "english"),
    ("The product looks nothing like the pictures, very disappointing", "shopping", "english"),
    ("This watch stopped working in two days, total waste of money", "shopping", "english"),
    ("quality is very poor, ek wash mein colour fade ho gaya", "shopping", "mixed"),
    ("product is not as shown, bilkul different hai", "shopping", "mixed"),
    ("size was wrong aur return bhi nahi ho raha", "shopping", "mixed"),
    ("stitching is terrible, pehli dafa mein phat gaya", "shopping", "mixed"),
    # delivery / service
    ("delivery 3 ghante late thi, koi call bhi nahi ki", "delivery", "roman_urdu"),
    ("rider bohat rude tha, order bhi galat de gaya", "delivery", "roman_urdu"),
    ("customer service ne meri baat hi nahi suni, bohat bura experience", "delivery", "roman_urdu"),
    ("order cancel kar diya bina bataye, bohat pareshani hui", "delivery", "roman_urdu"),
    ("staff ka behaviour bohat kharab tha, dobara nahi aaunga", "delivery", "roman_urdu"),
    ("Delivery took three hours and nobody even called to update me", "delivery", "english"),
    ("The rider was rude and delivered the wrong order", "delivery", "english"),
    ("Customer service ignored my complaint completely", "delivery", "english"),
    ("delivery was very late aur koi update bhi nahi di", "delivery", "mixed"),
    ("rider was rude, order bhi galat tha", "delivery", "mixed"),
    ("customer service was useless, meri problem solve nahi ki", "delivery", "mixed"),
    # mobile / app
    ("app har dafa crash ho jati hai, bilkul use nahi ho sakti", "mobile", "roman_urdu"),
    ("app bohat slow hai, order karna mushkil hai", "mobile", "roman_urdu"),
    ("new update ke baad app bilkul kharab ho gayi hai", "mobile", "roman_urdu"),
    ("login hi nahi ho raha, error aa raha hai", "mobile", "roman_urdu"),
    ("The app crashes every time I open it, completely unusable", "mobile", "english"),
    ("This app is so slow, it takes forever to load anything", "mobile", "english"),
    ("The latest update ruined the app, nothing works anymore", "mobile", "english"),
    ("app keeps crashing, bilkul use nahi ho sakti", "mobile", "mixed"),
    ("app is too slow, order karna bohat mushkil hai", "mobile", "mixed"),
    ("login nahi ho raha, error aa raha hai har dafa", "mobile", "mixed"),
]

NEUTRAL = [
    # food
    ("khana theek tha, na bohat acha na bohat bura", "food", "roman_urdu"),
    ("pizza normal thi, taste average tha", "food", "roman_urdu"),
    ("biryani theek thi, masalay thore kam thay", "food", "roman_urdu"),
    ("burger average tha, kuch khaas nahi tha", "food", "roman_urdu"),
    ("khana waqt par mila, taste normal tha", "food", "roman_urdu"),
    ("The food was okay, nothing special but not bad either", "food", "english"),
    ("Average pizza, the taste was fine but nothing memorable", "food", "english"),
    ("The biryani was alright, a bit less spicy than expected", "food", "english"),
    ("khana theek tha, taste was average", "food", "mixed"),
    ("pizza was okay, kuch khaas nahi tha", "food", "mixed"),
    ("burger was fine, average taste tha", "food", "mixed"),
    # shopping
    ("kapray ki quality theek hai, price ke hisaab se normal hai", "shopping", "roman_urdu"),
    ("shoes theek hain, na bohat comfortable na bohat kharab", "shopping", "roman_urdu"),
    ("product theek hai, photo jaisa hi hai", "shopping", "roman_urdu"),
    ("watch theek chal rahi hai, abhi tak koi problem nahi", "shopping", "roman_urdu"),
    ("The quality is fine for the price, nothing extraordinary", "shopping", "english"),
    ("Shoes are okay, they fit fine and look decent", "shopping", "english"),
    ("The product is as described, works as expected", "shopping", "english"),
    ("quality is okay, price ke hisaab se theek hai", "shopping", "mixed"),
    ("product is fine, kuch khaas nahi hai", "shopping", "mixed"),
    ("shoes are okay, fit theek hai", "shopping", "mixed"),
    # delivery / service
    ("delivery waqt par aa gayi, sab kuch theek tha", "delivery", "roman_urdu"),
    ("rider theek tha, order sahi mila", "delivery", "roman_urdu"),
    ("customer service ne jawab diya, problem abhi pending hai", "delivery", "roman_urdu"),
    ("staff normal tha, koi khaas baat nahi thi", "delivery", "roman_urdu"),
    ("Delivery arrived on time, everything was as ordered", "delivery", "english"),
    ("The rider was fine and the order was correct", "delivery", "english"),
    ("Customer service replied, my issue is still being looked into", "delivery", "english"),
    ("delivery was on time, sab kuch theek tha", "delivery", "mixed"),
    ("rider was fine, order sahi mila", "delivery", "mixed"),
    ("staff was okay, normal experience tha", "delivery", "mixed"),
    # mobile / app
    ("app theek chal rahi hai, koi badi problem nahi hai", "mobile", "roman_urdu"),
    ("app normal hai, kuch features slow hain", "mobile", "roman_urdu"),
    ("new update theek hai, kuch naya nahi hai", "mobile", "roman_urdu"),
    ("The app works fine, a few features are a bit slow", "mobile", "english"),
    ("The app is okay, nothing new in the latest update", "mobile", "english"),
    ("It does what it should, no major issues so far", "mobile", "english"),
    ("app is fine, kuch features slow hain", "mobile", "mixed"),
    ("app theek hai, koi badi problem nahi", "mobile", "mixed"),
    ("update was okay, kuch naya nahi tha", "mobile", "mixed"),
]

# Short follow-up clauses that make longer, realistic reviews when combined.
# The clause lists live in rusent.preprocessing (imported above) so the
# evaluation split can strip them back off and group near-duplicates.


def build_rows(sentences, sentiment, target):
    rows = []
    seen = set()
    pool = list(sentences)
    random.shuffle(pool)
    # First pass: every base sentence on its own.
    for text, topic, language in pool:
        if text not in seen:
            seen.add(text)
            rows.append((text, sentiment, topic, language))
    # Second pass: combine a base sentence with a follow-up clause until target.
    while len(rows) < target:
        base, topic, language = random.choice(sentences)
        follow = random.choice(FOLLOWUPS[sentiment])
        if language == "english":
            joiner = random.choice([". ", ", ", " and "])
            text = base.rstrip(".") + joiner + follow
        else:
            text = base.rstrip(".") + ", " + follow
        if text in seen:
            continue
        seen.add(text)
        rows.append((text, sentiment, topic, language))
    return rows[:target]


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "reviews.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += build_rows(POSITIVE, "positive", 140)
    rows += build_rows(NEGATIVE, "negative", 140)
    rows += build_rows(NEUTRAL, "neutral", 140)
    random.shuffle(rows)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "sentiment", "topic", "language"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} reviews to {out}")
    from collections import Counter

    print("sentiment:", Counter(r[1] for r in rows))
    print("topic:", Counter(r[2] for r in rows))
    print("language:", Counter(r[3] for r in rows))


if __name__ == "__main__":
    main()
