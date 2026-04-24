
import csv
import time
import requests

APP_ID = 1449110                   # CS:GO / CS2
TARGET_TOTAL = 3000                # total review yang ingin disimpan
POOL_SIZE = 9000                   # banyak review yang dikumpulkan dulu sebelum disampling
NUM_PER_PAGE = 100                 # max 100
LANGUAGE = "english"                 
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 1.5       # jeda biar tidak kena rate limit
OUTPUT_CSV = "steam_reviews_1449110.csv"

BASE_URL = "https://store.steampowered.com/appreviews/{appid}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_reviews_pool(app_id, pool_size):
    """Tarik review berurutan pakai cursor sampai mencapai pool_size atau habis."""
    reviews = []
    cursor = "*"
    seen_ids = set()
    page = 0

    while len(reviews) < pool_size:
        page += 1
        params = {
            "json": 1,
            "filter": "recent",            # urut dari terbaru -> terlama
            "language": LANGUAGE,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": NUM_PER_PAGE,
            "cursor": cursor,
        }

        url = BASE_URL.format(appid=app_id)
        try:
            resp = requests.get(url, params=params, headers=HEADERS,
                                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[page {page}] gagal ambil data: {e} -- coba lagi 5 detik")
            time.sleep(5)
            continue

        if data.get("success") != 1:
            print(f"[page {page}] respons tidak success, berhenti.")
            break

        batch = data.get("reviews", []) or []
        if not batch:
            print(f"[page {page}] tidak ada review lagi, berhenti.")
            break

        new_in_batch = 0
        for r in batch:
            rid = str(r.get("recommendationid"))
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            reviews.append(r)
            new_in_batch += 1

        next_cursor = data.get("cursor")
        print(f"[page {page}] +{new_in_batch} review (total {len(reviews)}), "
              f"cursor berikutnya: {next_cursor!r}")

        if not next_cursor or next_cursor == cursor:
            print("Cursor tidak berubah / kosong, berhenti.")
            break
        cursor = next_cursor

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return reviews


def sample_by_time_buckets(reviews, total):
    """Urutkan review by timestamp_created lalu ambil porsi seimbang dari
    bucket terlama, menengah, terbaru."""
    if not reviews:
        return []

    sorted_reviews = sorted(reviews, key=lambda r: r.get("timestamp_created", 0))
    n = len(sorted_reviews)

    third = n // 3
    oldest_bucket = sorted_reviews[:third]
    middle_bucket = sorted_reviews[third: 2 * third]
    newest_bucket = sorted_reviews[2 * third:]

    per_bucket = total // 3
    remainder = total - per_bucket * 3   # sisa diberikan ke bucket terbaru

    take_old = min(per_bucket, len(oldest_bucket))
    take_mid = min(per_bucket, len(middle_bucket))
    take_new = min(per_bucket + remainder, len(newest_bucket))

    def stride_pick(bucket, k):
        if k <= 0 or not bucket:
            return []
        if k >= len(bucket):
            return list(bucket)
        step = len(bucket) / k
        return [bucket[int(i * step)] for i in range(k)]

    picked = (
        stride_pick(oldest_bucket, take_old)
        + stride_pick(middle_bucket, take_mid)
        + stride_pick(newest_bucket, take_new)
    )

    # tambal kalau kurang
    if len(picked) < total:
        picked_ids = {str(r.get("recommendationid")) for r in picked}
        for r in sorted_reviews:
            if len(picked) >= total:
                break
            if str(r.get("recommendationid")) not in picked_ids:
                picked.append(r)

    return picked[:total]


def flatten_review(r):
    """Ubah satu review JSON jadi 1 baris flat untuk CSV. Tanpa preprocessing teks."""
    author = r.get("author", {}) or {}
    return {
        "recommendationid": r.get("recommendationid"),
        "steamid": author.get("steamid"),
        "num_games_owned": author.get("num_games_owned"),
        "num_reviews": author.get("num_reviews"),
        "playtime_forever_min": author.get("playtime_forever"),
        "playtime_last_two_weeks_min": author.get("playtime_last_two_weeks"),
        "playtime_at_review_min": author.get("playtime_at_review"),
        "last_played": author.get("last_played"),
        "language": r.get("language"),
        "review": r.get("review"),
        "timestamp_created": r.get("timestamp_created"),
        "timestamp_updated": r.get("timestamp_updated"),
        "voted_up": r.get("voted_up"),
        "votes_up": r.get("votes_up"),
        "votes_funny": r.get("votes_funny"),
        "weighted_vote_score": r.get("weighted_vote_score"),
        "comment_count": r.get("comment_count"),
        "steam_purchase": r.get("steam_purchase"),
        "received_for_free": r.get("received_for_free"),
        "written_during_early_access": r.get("written_during_early_access"),
        "hidden_in_steam_china": r.get("hidden_in_steam_china"),
        "steam_china_location": r.get("steam_china_location"),
    }


def save_to_csv(reviews, path):
    if not reviews:
        print("Tidak ada review untuk disimpan.")
        return

    rows = [flatten_review(r) for r in reviews]
    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Disimpan {len(rows)} review ke {path}")


def main():
    print(f"Mulai mengumpulkan review pool (target pool: {POOL_SIZE}) ...")
    pool = fetch_reviews_pool(APP_ID, POOL_SIZE)
    print(f"Total terkumpul di pool: {len(pool)}")

    sampled = sample_by_time_buckets(pool, TARGET_TOTAL)
    print(f"Total setelah sampling 3-bucket waktu: {len(sampled)}")

    save_to_csv(sampled, OUTPUT_CSV)


if __name__ == "__main__":
    main()