+++ 
draft = false
date = 2024-11-20T08:31:40+01:00
title = "2025 Year of Zig"
description = ""
slug = ""
authors = []
tags = ["zig"]
categories = []
externalLink = ""
series = []
+++


# Intro

Programming languages come and go and during my 20 years of coding I have used many of them to at least some degree (more than just hello world). Thanks to [ThePrimeagen](https://www.youtube.com/c/theprimeagen) I decided to take [Zig](https://ziglang.org/) for a spin. And boy I really like it! Just look at the mascot:

![](/images/zig_zero.png)

Come on, a crocodile with a jetpack? How cool is that?


# Why Zig

Sure I would not invest a lot of time into anything that a full time YouTuber recommends. However, I was in search of a highly performant programming language that compiles to native code, has excellent C compatibility, and most importantly, is fun to write code in. If I look back at my past nearly 10 years, the most dominant languages I used were Python and Scala with a detour to Go, and some necessary evil of Javascript/Typescript (Thank god this was minimal). For a long time I thought Scala was the pinnacle of programming languages, it had everything: Classes, Traits, Pattern Matching, For Comprehension, Destructuring, Options, Eithers, Try, Monads, Monoids, EitherT, OptionT, Applicatives, Functors, Kleisli, .... And the list goes on nearly forever. I felt extremely smart writing Scala, however that was also its demise. You can take 2 people with the same experience, and their Scala code could look like two different programming languages. And that is a serious issue - the last thing you need is a language that takes forever to onboard a new person to, especially if this person is already experienced with the language. After 5 years of full time Scala development, I was competent and comfortable, but was well aware that there was a lot to the language I did not know, and this feeling of constantly chasing mastery felt not really rewarding.

At some point I started to learn Go, which has a totally different philosophy compared to Scala - it is minimal and simple. I was able to learn the language, sure not master it, but I was able to somewhat efficiently use channels and structure my code nicely in a couple of days, and ship something to production not long after that. That was the point when I was sure I would never go back to Scala again.

Zig is actually very similar to Go. It is a simple language, ultimately fun to write in, and what's best, it is extremely performant.

# Little Tour of Zig

Let's look at a for loop in Zig,     

```zig

var array = [_]u32{ 1, 2, 3 };
for (array) |elem| {
    ...
    }
```

You also want to have the index of the array? Sure, here it is:
```zig
for (array, 0..) |elem, index| {
    ...
}
```

Your code can have exceptions? Sure, we have values as errors:
```zig
fn divide(q: u32, d:u32) error{ZeroDivision}!u32 {
  if (d == 0) {

    return error.ZeroDivision;
  }
  return q/d;
}


const result = divide(10, 0) catch |err| {
    // OOPS ERROR
    return ;

};


// OR better we can let it crash

const result = try divide(10,0);

```

What about Null? Well, we have Optional, with syntactic sugar for default cases:

```zig
const x: ?f32 = null;
const y = a orelse 0;
```


What if this optional is actually needed? Well, we can let it crash:

```zig
const a: ?f32 = 5;
const b = a orelse unreachable;
```

Unreachable tells the compiler to crash the program if we take an unreachable branch.


## Safety

Zig is a modern systems programming language, and since C++ is unsafe to use (especially by minors), it is important that the language provides features that ensure memory safety! Go achieves this by being garbage collected, while Rust uses the borrow checker and lifetimes. 

### Allocators
By default, if you define any variable in Zig it always lives on the Stack, however this may be limited - we can put things on the stack only if we know their size at compile time! To have objects that can dynamically grow, we have to put them on the Heap. To do this Zig provides allocators:

```zig

fn some_function() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    defer {
        const deinit_status = gpa.deinit();

        if (deinit_status == .leak) expect(false) catch @panic("I leaked memory");
    }
    var list = ArrayList(u8).init(allocator);
    defer list.deinit();
    try list.append('H');
    try list.append('e');
    try list.append('l');
    try list.append('l');
    try list.append('o');
    try list.appendSlice(" World!");
}
```

Here are 2 important parts to understand:
1. Use of defer keyword, which delays the execution of a block or function when the variable (block) leaves the scope. This sounds complicated - we can imagine that defer automatically moves the statement to the bottom of the function, so they are executed before the function ends regardless of where we define it!
2. Objects that leverage the heap always take an allocator, this follows the "no hidden allocations" philosophy of Zig, making heap use explicit (and also easy to audit). 

If we would compare this to Rust, if we want to define something on the heap we can use:
1. Rc, this is reference counting, and it is equivalent to shared_ptr of C++, for data that can be shared 
2. Arc, this is equivalent to unique_ptr of C++, this is for data that cannot be shared, has to have exactly one owner
3. Box, this will just put the object on the heap and the same ownership rules apply as to stack variables, where we can have exactly one mutable access and unlimited readonly const


```rust
let my_box = Box::new(1); 
```

So this will put the 1 on the heap and then the compiler is responsible to release the memory when my_box leaves the scope, if we look at the definition of Box:

```rust
pub struct Box<T, A = Global>
where
    A: Allocator,
    T: ?Sized;
```

We can see that Rust also has Allocators however they are implicit, this makes it less straightforward what object goes on the heap and what is on the stack, and the problem becomes even more obvious for the infamous str vs String:

```rust
let hello_world = "Hello, World!";
let hello = String::from("Hello, world!");
```

The first lives on the stack, and the second lives on a heap. Again this is something one will learn with time, but it is an implicit behavior making the language more complex.


### Arrays, Slices and Sentinels

If you ever wrote some C program, there is a good chance that you wrote it vulnerable. Why? Well by default C strings are null terminated, which means the length of a string is not really known, and it is extremely easy to leak data that we should not! Zig has two concepts that make things a bit more safe:

#### Arrays and Slices

Here we require that the arrays have a known size, and we are not really allowed to go beyond this size, otherwise we get an exception:

```zig
const a = [_]u8{ 'h', 'e', 'l', 'l', 'o' }; // this will end up as [5]u8
const slice = a[0..3];
const slice_to_te_end = a[0..];
a[4] // this blows up, this is an off by one error 
```

#### Sentinels
Using sentinels we can "extend" array by an extra value:

```zig
const ar = [_:0]u32{ 1, 2 };
assert(ar[2] == 0); // this is actually of by one but we get 0 since it is a sentinel
```

## C Interop

So you want some C code in your Zig project? Here is how you do it:

```zig
const elf = @cImport({
    @cInclude("elf.h");
});
```

Yep, @cImport and you are good to go, you can use functions and structs from in our case ```elf.h``` and you would use them like any other struct or function from Zig.

## Comptime

This is an extremely useful feature, which allows executing Zig during compilation to modify the source code itself:

```zig
pub fn ElfFile(comptime T: type) type {

    const is64Bit = comptime (@typeName(T) == @typeName(elf.Elf64_Ehdr));

    const S: type = comptime switch (@typeName(T)) {
        @typeName(elf.Elf64_Ehdr) => elf.Elf64_Shdr,
        @typeName(elf.Elf32_Ehdr) => elf.Elf32_Shdr,
        else => unreachable,
    };
    const P: type = comptime switch (@typeName(T)) {
        @typeName(elf.Elf64_Ehdr) => elf.Elf64_Phdr,
        @typeName(elf.Elf32_Ehdr) => elf.Elf32_Phdr,
        else => unreachable,
    };

    ...
}
```

The snippet above is from an ElfFile reader that I am writing for my [Dissecting Binaries](https://codebreakers.re/courses) course. It is actually used as:

```zig
const ElfReader = union(enum) {
    bit32: ElfFile(elf.Elf32_Ehdr),
    bit64: ElfFile(elf.Elf64_Ehdr),

    pub fn free(self: ElfReader) void {
        switch (self) {
            inline else => |case| return case.free(),
        }
    }

    pub fn print_header(self: ElfReader) void {
        switch (self) {
            inline else => |case| return case.print_header(),
        }
    }
};

const res: ElfReader = blk: {
    switch (e_ident[elf.EI_CLASS]) {
        elf.ELFCLASS64 => {
            const bit64 = try ElfFile(elf.Elf64_Ehdr).init(file, allocator);
            break :blk ElfReader{ .bit64 = bit64 };
        },
        elf.ELFCLASS32 => {
            const bit32 = try ElfFile(elf.Elf32_Ehdr).init(file, allocator);
            break :blk ElfReader{ .bit32 = bit32 };
        },
        else => unreachable,
    }
};
```

This is a long example, however it shows the power of comptime, which replaces generics and macros with just Zig! I remember trying to learn [Scala Macros](https://docs.scala-lang.org/overviews/macros/overview.html). While I understand that macros operate on the AST of the language, you still end up with an extra language embedded within a language.

## Recap

There are other language features I did not include, like Structs, Enums, Unions, ..., but they are nothing you have not seen in other languages. As I mentioned at the beginning, when you write Zig it feels very much like Go, but without garbage collection. If you add nice error handling, options instead of Nil, and Allocators for explicit memory management you end up with a language that is meant to be used not just in 2025, but also in many years to come. And on top of it, when you write Zig, you feel joy, which is really uncommon in programming languages lately (Especially Rust or TypeScript).

# Why Zig when there is Rust?

I guess you saw this coming. I have had a crush on Rust since 2014, however it never really took off. I did a couple of side projects, but I never really shipped anything to production. On paper Rust is the perfect language for me, it shares a lot of features with Scala, at least from the perspective of the type system and syntax. However, similarly to Scala, I always felt that Rust has just too many features, and I really need something that forces me to express what I want as simply as possible, but not too simple.

But there is more, I honestly believe that a systems programming language should not have automatic memory management. Why? There are multiple reasons. First, I believe that manual memory management is the key to unlock extra performance, and an example is the following [Zig vs Rust](https://www.youtube.com/watch?v=SR2LRhnL1AQ&t=69s&pp=ygUNemlnIGJlbmNobWFyaw%3D%3D) benchmark. Sure benchmarks are a lie, and I do not trust them even if they come from a third party. But since Zig is more low level than Rust, it gives you more control over your hardware, squeezing out some extra performance. 

The explicit use of Allocators gives me the possibility of implementing the following [SJMalloc](https://arxiv.org/pdf/2410.17928) allocator in Zig. Sure, I can do it for Rust as well, but then I have to find all the possible places which use an Allocator that are not explicitly defined, making it hard to integrate into an existing project.

And lastly, and probably the most important part for me, I am leveraging a lot of existing C libraries. For example, I am building a Fuzzer on top of QEMU. This is already done in [Rust LibAFL QEMU](https://www.s3.eurecom.fr/docs/bar24_malmain.pdf), however if you look at the code, you quickly realize that the authors had to first fork QEMU and implement extra features. Second, they needed to generate Rust bindings for the C code, and since QEMU is not really Rust, they extensively use unsafe. So rule of thumb, If you leverage a lot of C code, which I for most of my cybersecurity projects do, than Zig is your friend.

To recap, yes Rust is amazing when you stay in Rust, and there are a lot of things where the extra safety that Rust gives you is crucial. For example, SSL libraries or any library or service that is directly exposed to the internet, where any malicious use can have huge impact, should probably use Rust. But for the rest, there is Go, yes Go - this is not a typo, I wrote and meant Go! Zig is really perfect if you need to integrate with existing C libraries and when you need to squeeze out every last drop of performance from your hardware. (Sure there is high-performance Rust, but that's another story, making Rust a whole different language, once again!). But most Web APIs, CLI tools can run for a couple of milliseconds longer, consuming extra couple of bytes memory.

## Is there More? 

Sure, Zig is an amazing build system, it can be used to build C/C++ projects and it supports cross compilation, which means you can emit for example ARM64 code on your x86 Linux Machine. Cross compilation makes it especially compelling for tools like [Bun](https://bun.sh/) and look up this [Bun vs Deno](https://www.youtube.com/watch?v=yJmyYosyDDM&t=459s&pp=ygULYnVuIHZzIGRlbm8%3D) benchmark, yes I know, benchmarks lie!
