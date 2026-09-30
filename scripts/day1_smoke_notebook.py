# Databricks notebook source
"""Tiny synthetic serverless batch computation for Day 1 capability probing."""

import json

from pyspark.sql import functions as F


summary = spark.range(4).agg(F.count("id").alias("n"), F.sum("id").alias("sum_id")).first()
assert summary.n == 4 and summary.sum_id == 6
dbutils.notebook.exit(json.dumps({"synthetic_rows": summary.n, "sum_id": summary.sum_id}))
