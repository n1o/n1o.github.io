+++ 
draft = true
date = 2021-03-26T16:08:17+01:00
title = "Automate BigQuery using Terraform"
description = ""
slug = ""
authors = []
tags = ["bigquery", "terraform"]
categories = []
externalLink = ""
series = []
+++

Keeping table schemas up to date in multiple projects across different environments is a challenge. The same applies to [BigQuery](https://cloud.google.com/bigquery/). For those of you that do know, BigQuery is a serverless data warehouse on [Google cloud platform](https://cloud.google.com/). Lets imagine that we have three environments:

1. Production
2. Staging
3. Development

If we would like to add a new column to one our tables, we have multiple options:

* Use the web based user interface

* Use the CLI tool

* Use [Terraform](https://www.terraform.io/)

Using the web user interface is error prone, and if we have to manage multiple environments which can become tedious. The CLI does a good job, but writing scripts can be hard, especially if we want them to be idempotent. The last an my favorite option is Terraform, with it we can easily automate creating datasets, tables and views. 

A dataset is an collection of tables, views and user defined functions. We can create a dataset as follows:

```
resource "google_bigquery_dataset" "movies" {
  dataset_id                  = "movies"
  friendly_name               = "movies warehouse"
  description                 = "movies DWH"
  location                    = var.datasetLocation
}
```
The most important field is **dataset_id**, and it is similar to schema in an traditional relational database.

After we defined a dataset we can create some tables:

````
resource "google_bigquery_table" "movies" {
  dataset_id = google_bigquery_dataset.movies.dataset_id
  table_id   = "movies"
  schema = file("./schemas/movies.json")
}

resource "google_bigquery_table" "ratings" {
  dataset_id = google_bigquery_dataset.movies.dataset_id
  table_id   = "ratings"
  schema = file("./schemas/ratings.json")
}
````

In each table we specify:

* **dataset_id** is the reference to the dataset we want the table to exist
* **table_id** is the table name
* **schema** are the columns in the table

My personal preference is to keep the schema in separate files. This way the code is more readable, and can be reused by different tools.  For example [apache beam](https://beam.apache.org) or the official BigQuery SDK, require schema definitions when writing into tables.

Views are an important part of any warehouse. We can define a view as follows:

```
resource "google_bigquery_table" "view_average_raiting_per_genere" {
  dataset_id = google_bigquery_dataset.movies.dataset_id
  table_id   = "view_average_raiting_per_genere"
    view  {
      query = file("./views/view_average_raiting_per_genere.sql")
      use_legacy_sql = false
  }
  depends_on = [google_bigquery_table.ratings, google_bigquery_table.movies ]
}
```

The fields are similar to tables, but with the distinction that we have to provide the query for the view instead of a schema. The query is just an select statement, and as with table schemas I prefer to keep them in separate files. Since most views depend on tables, we can model the dependency by listing the dependencies in the depends_on attribute. This way we can keep track of which view is using what table. If we would forget this, than a view could be created before the tables and we would get an error. 

This sound like a lot of work for creating a couple of tables and views. And you may ask why this is better using the CLI or even the web interface. The answer is simple, if we would like to modify a existing table, we just have to add a new field into the schema and run ```terraform apply```, or if we have CI/CD set up we can execute changes on every merge in our main branch. The same applies to views, if a view is changed we just simply update the sql script. 

Lets talk about the bad parts. Well if you decide to drop an column, than terraform wont help you and you have to do it manually. Luckily it is more common to ad new columns than removing old ones.

The biggest downside is handling user defined functions. 
