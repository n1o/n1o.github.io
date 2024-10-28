+++ 
draft = true
date = 2024-10-23T10:23:52+02:00
title = "Two Weeks of Zig"
description = ""
slug = ""
authors = []
tags = ["zig"]
categories = []
externalLink = ""
series = []
+++

# Why not Rust? 

I am already learning Rust from 2014, that is more than 10 years! During that time I wrote professionally code in multiple languages, including Scala, Python, Typescript, Go, C++, Java, Elixir, Clojure and lately x86 assembly. For me Rust never really took of. On paper it is quite similar to Scala, you get Options, patter machine, Errors as return values, and boy I really loved coding Scala, but i ditched it in favour of Python an Go.

## A little bit about my Scala Journey

During the early 2010s, as many of us I was primarily paid to ship Java. Back then that was Java 1.5 and 1.6, the standard for many years to come. A lot of has changes since then, or more precisely there where a lot of different things trendy since then. To clarify I was in my early 20s, and increadably biased towards the hot cool stuf. One of the trends was the surge of functional programming. Everybody was trying to [learn Haskel](https://learnyouahaskell.com/) and the JVM communityy was all about Clojure and Scala. You can already guess, I stareted to learn Scala by picking up [this book!](https://www.goodreads.com/book/show/5680904-programming-in-scala), yes people used to learn coding from books! And I was stunned, the promise of immutability, functional composition, algebraic types and boy pattern matching! I was in love. But love is also blinding, I did a great sin. I started to heavily push functional programming into an ancient Spring web applications running Java 1.5 (eventually I migrated it to 1.6). Since Java 1.5 gave little room (literally none) for functional programming I went and included [Guava](https://github.com/google/guava) and starated to produce immutable piles of functional sh*tty code, and I loved, but back than I was the solo developer on the team. But things change, and I was about to leave the project and as it turns out the company as well, and I got a new colleague. The handower of my shitty, and in hindsight unnecessary functional mess was less than optimal. Of course back than, I attributed my colleagues strougles as skill issues, but now after nearly 15 years I can learly see the obvious signs of hype driven development. 

I got a new shiny job at a startup, and my job was coding Scala full time, and it started my next 5 years of continually improving, becoming more Haskell like with [ScalaZ](https://github.com/scalaz/scalazhttps://github.com/scalaz/scalaz) and later with [Cats](https://typelevel.org/cats/). Scala and Rust share a lot of language features, and I argue anybody who wrote Scala long enough, can get profficient in basic Rust in little time. Yes there is borrow checker, but the ideas is simple, technically at any given time you can have exactly one owner, where the ownership can be transferred. Lifetimes a bit more tricky, but again, if you are using an object, it has to live at least as long as you.You got Box which is unique pointer, Rc which is reference counting, Arc is thread safe Rc. Macros are just by default confusing in booth cases, I implemented a couple of marcos in Scala, but none in Rust. And there is async, in Scala you work with Futures and you have for expressions as a syntacticc suggar to chain them, where in Rust it feels way more like Javascript. I like the convince of async/await of Javascript, however I thing it is misplaced in Rust or potentially in any systems level programming language. Why missplaced? The promise of rust is safe memory management, this means the compiler is responsible to free unused memory, to make this possible you have to introduce rules what can and cannot compile. This is done trough the borrow checker and lifetimes. What is async? Async is a computation (or IO or whatever) that will be completed sometime in the future. Combining strict rules for memory managedment with the vague promises of future computation introduces a lot of hardship. Especially for people coming from other garbage collected language (everything is just skill issues).


Back to Scala, after 5 years of proffesinally writing Scala every day, I was feeling profficient, and I did feel smart, but I also felt that mastring it would take ages and I was increadably far away. This feeling of continuously learning, but never achieving mastery was one of the things that in the long run pushed me away from the language. And my feeling, that Rust has the same trajectory. The more features, the more ways to achieve the same things is in a given programming language, the harder is the eventual mastery.  However Rust has its place, It is a tool to write an modern Operating System, for system libraries like SSL, SSH, WEBP or compression. I would even say that Rust is the right tool for every piece of software, but probably you should not use it.

## Enter Go

It is a strange thing to start writing about Zig by first mentioning its arch nemesis Rust, and than continue with is life-long rival Go. At first in my Scala days, I did not like Go very much, and even after I abandoned Scala and wrote mostly just Python, i still shuned Go as an inferior language, at least syntax wise.

I remember when I started to learn Go, I completed [Go By example](https://gobyexample.com/) and in two hours I was ready to ship it to production. No kidding, at that time I was a proffesinal developer for nearly a decade. I spend my last year mostly writing Python, and It became my go-to language. Not because I was the fastest, and even today, nearly after 12 years since I wrote my first Python script, I still hardly consider myself as an expert, but again I was doing mostly Machine Learning and some data crunching, there is just so much Python I really need. There is a lot of similarity between Python and Go, (No shit it was design that way!), but what I started to appreciate the most are its simplicity. 

# Why Zig

https://www.youtube.com/watch?v=yJmyYosyDDM

There are a lot of similarties between Zig and Go, I did complete the [Ziglings](https://codeberg.org/ziglings/exercises/) in a couple of lunch breaks. An I feel confident enough to ship it. Again that is mainly due the simplicity of the language. 
