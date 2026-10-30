#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from sql_reasoner.formats import format_model_output
from sql_reasoner.io import write_jsonl
from sql_reasoner.planning import estimate_difficulty, plan_from_sql
from sql_reasoner.prompts import messages_for_sft
from sql_reasoner.sandbox import execute_read_only
from sql_reasoner.schema import introspect_schema, serialize_schema


CASES = [
    ("客户总数是多少？", "SELECT COUNT(*) FROM customers"),
    ("目前有多少种商品？", "SELECT COUNT(*) FROM products"),
    (
        "所有已完成订单的销售额是多少？",
        "SELECT ROUND(SUM(total_amount), 2) FROM orders WHERE status = 'completed'",
    ),
    (
        "2025年1月一共下了多少笔订单？",
        "SELECT COUNT(*) FROM orders WHERE order_date >= '2025-01-01' AND order_date < '2025-02-01'",
    ),
    (
        "列出上海客户的姓名和会员等级。",
        "SELECT name, membership_level FROM customers WHERE city = '上海' ORDER BY name",
    ),
    (
        "库存少于20件的商品有哪些？",
        "SELECT name, stock FROM products WHERE stock < 20 ORDER BY stock, name",
    ),
    (
        "单价最高的5种商品是什么？",
        "SELECT name, price FROM products ORDER BY price DESC, name LIMIT 5",
    ),
    ("平均商品单价是多少？", "SELECT ROUND(AVG(price), 2) FROM products"),
    (
        "每个会员等级分别有多少客户？",
        "SELECT membership_level, COUNT(*) FROM customers GROUP BY membership_level ORDER BY membership_level",
    ),
    (
        "各城市客户数量是多少？",
        "SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY COUNT(*) DESC, city",
    ),
    (
        "每个订单状态有多少笔订单？",
        "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY status",
    ),
    (
        "每家门店完成订单的销售额是多少？",
        "SELECT s.name, ROUND(SUM(o.total_amount), 2) FROM stores s JOIN orders o ON s.id = o.store_id WHERE o.status = 'completed' GROUP BY s.id, s.name ORDER BY SUM(o.total_amount) DESC",
    ),
    (
        "购买次数最多的5位客户是谁？",
        "SELECT c.name, COUNT(*) AS order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY order_count DESC, c.name LIMIT 5",
    ),
    (
        "销售数量最多的5种商品是什么？",
        "SELECT p.name, SUM(oi.quantity) AS units FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY units DESC, p.name LIMIT 5",
    ),
    (
        "各商品类别的已完成销售额是多少？",
        "SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM categories c JOIN products p ON c.id = p.category_id JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY c.id, c.name ORDER BY revenue DESC",
    ),
    (
        "哪些客户从未下过订单？",
        "SELECT c.name FROM customers c LEFT JOIN orders o ON c.id = o.customer_id WHERE o.id IS NULL ORDER BY c.name",
    ),
    (
        "哪些商品从未被购买？",
        "SELECT p.name FROM products p LEFT JOIN order_items oi ON p.id = oi.product_id WHERE oi.id IS NULL ORDER BY p.name",
    ),
    (
        "订单金额高于平均订单金额的订单编号和金额有哪些？",
        "SELECT id, total_amount FROM orders WHERE total_amount > (SELECT AVG(total_amount) FROM orders) ORDER BY total_amount DESC, id",
    ),
    (
        "消费总额高于所有客户平均消费总额的客户有哪些？",
        "WITH customer_spend AS (SELECT c.id, c.name, COALESCE(SUM(o.total_amount), 0) AS spend FROM customers c LEFT JOIN orders o ON c.id = o.customer_id AND o.status = 'completed' GROUP BY c.id, c.name) SELECT name, ROUND(spend, 2) FROM customer_spend WHERE spend > (SELECT AVG(spend) FROM customer_spend) ORDER BY spend DESC, name",
    ),
    (
        "每位客户最近一次下单日期是什么？",
        "SELECT c.name, MAX(o.order_date) FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY c.name",
    ),
    (
        "2025年每个月的已完成销售额是多少？",
        "SELECT SUBSTR(order_date, 1, 7) AS month, ROUND(SUM(total_amount), 2) FROM orders WHERE status = 'completed' AND order_date >= '2025-01-01' AND order_date < '2026-01-01' GROUP BY month ORDER BY month",
    ),
    (
        "各门店的平均客单价是多少？",
        "SELECT s.name, ROUND(AVG(o.total_amount), 2) FROM stores s JOIN orders o ON s.id = o.store_id WHERE o.status = 'completed' GROUP BY s.id, s.name ORDER BY s.name",
    ),
    (
        "退货次数最多的商品类别是什么？",
        "SELECT c.name, COUNT(*) AS return_count FROM returns r JOIN order_items oi ON r.order_item_id = oi.id JOIN products p ON oi.product_id = p.id JOIN categories c ON p.category_id = c.id GROUP BY c.id, c.name ORDER BY return_count DESC, c.name LIMIT 1",
    ),
    ("退款总金额是多少？", "SELECT ROUND(SUM(refund_amount), 2) FROM returns"),
    (
        "每种退货原因分别有多少次？",
        "SELECT reason, COUNT(*) FROM returns GROUP BY reason ORDER BY COUNT(*) DESC, reason",
    ),
    (
        "有退货记录的客户有哪些？",
        "SELECT DISTINCT c.name FROM customers c JOIN orders o ON c.id = o.customer_id JOIN order_items oi ON o.id = oi.order_id JOIN returns r ON oi.id = r.order_item_id ORDER BY c.name",
    ),
    (
        "同时购买过至少3种不同商品的客户有哪些？",
        "SELECT c.name, COUNT(DISTINCT oi.product_id) AS product_kinds FROM customers c JOIN orders o ON c.id = o.customer_id JOIN order_items oi ON o.id = oi.order_id GROUP BY c.id, c.name HAVING COUNT(DISTINCT oi.product_id) >= 3 ORDER BY product_kinds DESC, c.name",
    ),
    (
        "完成订单数至少为2的门店有哪些？",
        "SELECT s.name, COUNT(*) AS completed_orders FROM stores s JOIN orders o ON s.id = o.store_id WHERE o.status = 'completed' GROUP BY s.id, s.name HAVING COUNT(*) >= 2 ORDER BY completed_orders DESC, s.name",
    ),
    (
        "每个类别中价格最高的商品有哪些？",
        "SELECT c.name, p.name, p.price FROM categories c JOIN products p ON c.id = p.category_id WHERE p.price = (SELECT MAX(p2.price) FROM products p2 WHERE p2.category_id = c.id) ORDER BY c.name, p.name",
    ),
    (
        "每个城市已完成订单的销售额是多少？",
        "SELECT c.city, ROUND(SUM(o.total_amount), 2) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.city ORDER BY SUM(o.total_amount) DESC, c.city",
    ),
    (
        "金卡客户的已完成订单平均金额是多少？",
        "SELECT ROUND(AVG(o.total_amount), 2) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.membership_level = '金卡' AND o.status = 'completed'",
    ),
    (
        "北京门店卖出了多少件商品？",
        "SELECT SUM(oi.quantity) FROM stores s JOIN orders o ON s.id = o.store_id JOIN order_items oi ON o.id = oi.order_id WHERE s.city = '北京' AND o.status = 'completed'",
    ),
    (
        "折扣率大于10%的订单明细有多少条？",
        "SELECT COUNT(*) FROM order_items WHERE discount_rate > 0.10",
    ),
    (
        "没有使用折扣的商品销售额是多少？",
        "SELECT ROUND(SUM(quantity * unit_price), 2) FROM order_items WHERE discount_rate = 0",
    ),
    (
        "每种商品的实际销售额是多少？",
        "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount_rate)), 2) AS net_revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON o.id = oi.order_id WHERE o.status = 'completed' GROUP BY p.id, p.name ORDER BY net_revenue DESC, p.name",
    ),
    (
        "2025年第二季度完成了多少笔订单？",
        "SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date >= '2025-04-01' AND order_date < '2025-07-01'",
    ),
    (
        "注册时间最早的3位客户是谁？",
        "SELECT name, registered_at FROM customers ORDER BY registered_at, name LIMIT 3",
    ),
    (
        "每家门店有多少名不同客户下过单？",
        "SELECT s.name, COUNT(DISTINCT o.customer_id) FROM stores s LEFT JOIN orders o ON s.id = o.store_id GROUP BY s.id, s.name ORDER BY s.name",
    ),
    (
        "只下过一次订单的客户有哪些？",
        "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name HAVING COUNT(*) = 1 ORDER BY c.name",
    ),
    (
        "订单数最多的城市是什么？",
        "SELECT c.city, COUNT(*) AS order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.city ORDER BY order_count DESC, c.city LIMIT 1",
    ),
    (
        "价格在100元到500元之间的商品有哪些？",
        "SELECT name, price FROM products WHERE price BETWEEN 100 AND 500 ORDER BY price, name",
    ),
    (
        "名称中包含“Pro”的商品有哪些？",
        "SELECT name FROM products WHERE name LIKE '%Pro%' ORDER BY name",
    ),
    (
        "各类别当前总库存是多少？",
        "SELECT c.name, SUM(p.stock) FROM categories c JOIN products p ON c.id = p.category_id GROUP BY c.id, c.name ORDER BY c.name",
    ),
    (
        "总库存最高的商品类别是什么？",
        "SELECT c.name, SUM(p.stock) AS total_stock FROM categories c JOIN products p ON c.id = p.category_id GROUP BY c.id, c.name ORDER BY total_stock DESC, c.name LIMIT 1",
    ),
    (
        "退货金额占已完成销售额的比例是多少？",
        "SELECT ROUND((SELECT SUM(refund_amount) FROM returns) / (SELECT SUM(total_amount) FROM orders WHERE status = 'completed'), 4)",
    ),
    (
        "没有发生退货的已完成订单有多少笔？",
        "SELECT COUNT(*) FROM orders o WHERE o.status = 'completed' AND NOT EXISTS (SELECT 1 FROM order_items oi JOIN returns r ON oi.id = r.order_item_id WHERE oi.order_id = o.id)",
    ),
    (
        "至少在两家不同门店消费过的客户有哪些？",
        "SELECT c.name, COUNT(DISTINCT o.store_id) AS store_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name HAVING COUNT(DISTINCT o.store_id) >= 2 ORDER BY store_count DESC, c.name",
    ),
    (
        "每位客户完成订单和取消订单的数量分别是多少？",
        "SELECT c.name, SUM(CASE WHEN o.status = 'completed' THEN 1 ELSE 0 END) AS completed_count, SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_count FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY c.name",
    ),
    (
        "销售额排名第二的门店是哪家？",
        "SELECT s.name, ROUND(SUM(o.total_amount), 2) AS revenue FROM stores s JOIN orders o ON s.id = o.store_id WHERE o.status = 'completed' GROUP BY s.id, s.name ORDER BY revenue DESC, s.name LIMIT 1 OFFSET 1",
    ),
    (
        "每种会员等级的平均累计消费是多少？",
        "WITH customer_spend AS (SELECT c.id, c.membership_level, COALESCE(SUM(CASE WHEN o.status = 'completed' THEN o.total_amount ELSE 0 END), 0) AS spend FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.membership_level) SELECT membership_level, ROUND(AVG(spend), 2) FROM customer_spend GROUP BY membership_level ORDER BY membership_level",
    ),
]

PARAPHRASES = [
    "咱们一共有多少客户？",
    "商品SKU总数呢？",
    "完成单总共卖了多少钱？",
    "一月份有几单？",
    "上海的客户叫什么、分别是什么等级？",
    "哪些货快没库存了，按20件算？",
    "最贵的五个商品列一下。",
    "商品均价多少？",
    "按会员等级统计人数。",
    "每座城市各有几位客户？",
    "订单状态分别有几单？",
    "各店完成单GMV是多少？",
    "下单最频繁的五位客户。",
    "按销量找前五商品。",
    "各品类完成单收入。",
    "从来没买过东西的客户。",
    "还没有卖出去过的商品。",
    "找出金额超过订单均值的单子。",
    "哪些客户消费比人均消费高？",
    "客户最近啥时候下的单？",
    "按月看2025年的完成单销售额。",
    "每家店完成订单的客单价。",
    "退货最多的是哪个品类？",
    "一共退了多少钱？",
    "不同退货原因各几次？",
    "哪些人退过货？",
    "买过至少三种商品的客户。",
    "完成单不少于两笔的店。",
    "每个分类里最贵的商品。",
    "按客户城市统计完成单收入。",
]


def create_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(20251026)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL, membership_level TEXT NOT NULL, registered_at TEXT NOT NULL);
            CREATE TABLE stores (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT NOT NULL);
            CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
            CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category_id INTEGER NOT NULL REFERENCES categories(id), price REAL NOT NULL, stock INTEGER NOT NULL);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id), store_id INTEGER NOT NULL REFERENCES stores(id), order_date TEXT NOT NULL, status TEXT NOT NULL, total_amount REAL NOT NULL);
            CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(id), product_id INTEGER NOT NULL REFERENCES products(id), quantity INTEGER NOT NULL, unit_price REAL NOT NULL, discount_rate REAL NOT NULL DEFAULT 0);
            CREATE TABLE returns (id INTEGER PRIMARY KEY, order_item_id INTEGER NOT NULL REFERENCES order_items(id), return_date TEXT NOT NULL, reason TEXT NOT NULL, refund_amount REAL NOT NULL);
            """
        )
        cities = ["北京", "上海", "深圳", "杭州", "成都"]
        levels = ["普通", "银卡", "金卡"]
        conn.executemany(
            "INSERT INTO stores VALUES (?, ?, ?)",
            [(i + 1, f"{city}中心店", city) for i, city in enumerate(cities)],
        )
        conn.executemany(
            "INSERT INTO categories VALUES (?, ?)",
            [(1, "电脑"), (2, "手机"), (3, "家电"), (4, "配件")],
        )
        products = []
        for product_id in range(1, 25):
            category_id = (product_id - 1) % 4 + 1
            suffix = "Pro" if product_id % 5 == 0 else "标准版"
            products.append(
                (
                    product_id,
                    f"商品{product_id}-{suffix}",
                    category_id,
                    float(49 + product_id * 37),
                    (product_id * 11) % 80,
                )
            )
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", products)
        customers = []
        for customer_id in range(1, 31):
            registered = date(2023, 1, 1) + timedelta(days=customer_id * 19)
            customers.append(
                (
                    customer_id,
                    f"客户{customer_id:02d}",
                    cities[customer_id % len(cities)],
                    levels[customer_id % len(levels)],
                    registered.isoformat(),
                )
            )
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers)

        item_id = 1
        return_id = 1
        for order_id in range(1, 91):
            customer_id = rng.randint(1, 27)  # leave three customers without orders
            store_id = rng.randint(1, 5)
            order_date = date(2025, 1, 1) + timedelta(days=rng.randint(0, 300))
            status = rng.choices(["completed", "cancelled", "pending"], weights=[8, 1, 1])[0]
            item_rows = []
            total = 0.0
            for _ in range(rng.randint(1, 4)):
                product_id = rng.randint(1, 22)  # leave two products unsold
                quantity = rng.randint(1, 3)
                unit_price = products[product_id - 1][3]
                discount = rng.choice([0.0, 0.0, 0.05, 0.10, 0.15])
                item_rows.append((item_id, order_id, product_id, quantity, unit_price, discount))
                total += quantity * unit_price * (1 - discount)
                item_id += 1
            conn.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
                (order_id, customer_id, store_id, order_date.isoformat(), status, round(total, 2)),
            )
            conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)", item_rows)
            if status == "completed" and order_id % 11 == 0:
                chosen = item_rows[0]
                conn.execute(
                    "INSERT INTO returns VALUES (?, ?, ?, ?, ?)",
                    (
                        return_id,
                        chosen[0],
                        (order_date + timedelta(days=7)).isoformat(),
                        rng.choice(["质量问题", "不合适", "误购"]),
                        round(chosen[3] * chosen[4] * 0.8, 2),
                    ),
                )
                return_id += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/benchmarks/retail_cn")
    args = parser.parse_args()
    output = Path(args.output_dir)
    db_path = output / "retail.sqlite"
    create_database(db_path)
    schema = introspect_schema(db_path)
    schema_text = serialize_schema(schema)
    rows = []
    for index, (question, sql) in enumerate(CASES, start=1):
        execute_read_only(db_path, sql)
        plan = plan_from_sql(sql)
        assistant = format_model_output(plan, sql)
        row = {
            "id": f"retail-{index:03d}",
            "db_id": "retail_cn_private",
            "question": question,
            "schema": schema_text,
            "difficulty": estimate_difficulty(sql),
            "gold_plan": plan,
            "gold_sql": sql,
            "db_path": str(db_path),
            "split": "private_test",
            **messages_for_sft(question, schema_text, assistant),
        }
        rows.append(row)
    robust = []
    for index, question in enumerate(PARAPHRASES, start=1):
        source = rows[index - 1]
        robust.append(
            {
                **source,
                "id": f"retail-robust-{index:03d}",
                "question": question,
                "split": "robustness",
                "prompt": messages_for_sft(
                    question, schema_text, source["completion"][0]["content"]
                )["prompt"],
            }
        )
    write_jsonl(output / "test.jsonl", rows)
    write_jsonl(output / "robustness.jsonl", robust)
    print(f"created {db_path}; test={len(rows)} robustness={len(robust)}")


if __name__ == "__main__":
    main()
