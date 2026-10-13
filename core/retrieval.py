"""
Hybrid Product Retrieval System for VIKMO DealerDesk.
Combines exact token/attribute extraction (vehicle fitment, category, brand, SKU)
with TF-IDF character and word n-gram semantic vector similarity across the 600 catalogue items.
"""
import re
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.models import Product

_VECTORIZER = None
_PRODUCT_MATRIX = None
_PRODUCTS_CACHE: list[dict[str, Any]] = []
_SKU_INDEX: dict[str, int] = {}


def build_or_get_index():
    global _VECTORIZER, _PRODUCT_MATRIX, _PRODUCTS_CACHE, _SKU_INDEX
    if _VECTORIZER is not None and _PRODUCT_MATRIX is not None and _PRODUCTS_CACHE:
        return _VECTORIZER, _PRODUCT_MATRIX, _PRODUCTS_CACHE, _SKU_INDEX

    products = list(
        Product.objects.select_related("inventory").all().order_by("sku")
    )
    if not products:
        return None, None, [], {}

    cache = []
    corpus = []
    sku_idx = {}

    for idx, p in enumerate(products):
        doc = f"{p.sku} {p.name} {p.brand} {p.category} {p.vehicle_fitment} {p.description}"
        cache.append(
            {
                "sku": p.sku,
                "name": p.name,
                "category": p.category,
                "brand": p.brand,
                "vehicle_fitment": p.vehicle_fitment,
                "price": str(p.price),
                "stock": p.stock,
                "description": p.description,
            }
        )
        corpus.append(doc)
        sku_idx[p.sku] = idx

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer="word",
        token_pattern=r"(?u)\b[\w-]+\b",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)

    _VECTORIZER = vectorizer
    _PRODUCT_MATRIX = matrix
    _PRODUCTS_CACHE = cache
    _SKU_INDEX = sku_idx
    return _VECTORIZER, _PRODUCT_MATRIX, _PRODUCTS_CACHE, _SKU_INDEX


def invalidate_index():
    global _VECTORIZER, _PRODUCT_MATRIX, _PRODUCTS_CACHE, _SKU_INDEX
    _VECTORIZER = None
    _PRODUCT_MATRIX = None
    _PRODUCTS_CACHE = []
    _SKU_INDEX = {}


def retrieve_products(
    query: str,
    vehicle: str | None = None,
    category: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieves products matching query, vehicle, and optional category.
    Applies attribute boosting and vector similarity ranking.
    """
    vectorizer, matrix, cache, _ = build_or_get_index()
    if not cache or vectorizer is None:
        return []

    # Direct SKU check
    clean_q = query.strip().upper()
    direct_sku = [p for p in cache if clean_q == p["sku"]]
    if direct_sku:
        return direct_sku

    query_vec = vectorizer.transform([f"{query} {vehicle or ''} {category or ''}"])
    similarities = cosine_similarity(query_vec, matrix)[0]

    scored = []
    v_clean = vehicle.strip().lower() if vehicle else ""
    cat_clean = category.strip().lower() if category else ""
    q_tokens = set(re.findall(r"\w+", query.lower()))

    for i, p in enumerate(cache):
        sim = float(similarities[i])
        fitment_lower = p["vehicle_fitment"].lower()
        name_lower = p["name"].lower()
        brand_lower = p["brand"].lower()
        cat_lower = p["category"].lower()

        boost = 0.0

        # Vehicle fitment match
        if v_clean:
            if v_clean in fitment_lower or fitment_lower == "universal":
                boost += 0.4
            elif fitment_lower in v_clean:
                boost += 0.3
            else:
                boost -= 0.2

        # Category match
        if cat_clean and cat_clean == cat_lower:
            boost += 0.2

        # Token overlap in name
        overlap = len(q_tokens.intersection(set(re.findall(r"\w+", name_lower))))
        boost += overlap * 0.05

        total_score = sim + boost
        scored.append((total_score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]
