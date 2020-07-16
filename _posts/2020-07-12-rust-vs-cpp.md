---
toc: false
layout: post
description: Choosing a language to implement DL algorithms from scratch
categories: [markdown]
comments: true
image: images/rustvscpp.jpg
title: Rust v/s C++ for Deep learning
---

Hey there! 👋
<br>
Today, let's discuss about Rust v/s C++. Actually, it's not "Rust v/s C++" as in comparison of the languages, but I just want a language for understanding the underpinnings of deep learning algorithms. I want to build them from scratch or atleast try, so that I have a better understanding. One of my favorite physicists, [Richard Feynman](https://en.wikipedia.org/wiki/Richard_Feynman) said something I resonate with a lot:
> *"What I cannot create, I do not understand."*

 So, my search for a language begun. Let me tell you this, I really love **Python**. It's one language I really am very comfortable with and Python was what I used to get into data science (not that there were many others). Python was simple and easy. However, I wanted to do some low level programming. Enter **C++**. C was my first language way back in school, then C++ & Java. However, it has been years since I actually did something worthwhile with them. I did use Java in my undergrad but it wasn't as much. Then came Python which I learned as I wanted to get into DS, ML. And I fell in love!. After working with python for many months, one day I discovered that Python is not what is used in deployment! I was like what? really? So, then what? 

 I found C++ was what most production systems used. In my CS undergrad we had a subject called **Systems programming** in which we wrote assembly code on IBM 360. I really liked it, it was a pain in the ass to do things as simple as addition, substraction but it really made me understand how the code gets compiled and the internals of it. This started a love with low level programming, I always wanted to do build things from scratch but never got around to do it. It is something I wish I had done! I guess I never thought I could actually do it. But, life is too short for it. Now is the time I finally think of doing it.  

 Okay, all that aside, I now have narrowed down my choices to Rust & C++. There are other like Swift and Julia. Maybe Julia is more suited to this. But as I said I want to do it with a low level programming language. It would be shit-hard, but I want to challenge myself and if I can do atleast half of it, this would increase my confidence ten-fold. [Sundar Pichai](https://en.wikipedia.org/wiki/Sundar_Pichai) in one of his interview said, 
 > *"Try solving the hardest problem first, if you can then the easier ones will follow"*.   

 I did extensive research and found: If I use C++, there's a job market out there where I can transfer my C++ skills. On the other hand Rust is a pretty young language. But the thing is I gravitate more to Rust & the heart wants what it wants. Naturally, I went through all the blogs and reddit. There were two groups of people. The first kind termed rust as a revolution, as something they called a **memory safe** language and gave C++ a lot of flak. The second ones said it's never gonna replace C & C++ anytime soon, pretty correct actually since a lot of code bases is all C & C++. But, code safety with what I was able to understand is basically just bad code. Rust provides a stricter compliler check like you can't simple pass values around. So what do I do?

 I will take my chance & go with Rust ❤

As always, thank you for reading 😊😃!


# (UPDATE)

Wasn't as simple as I thought 😅

This is an update on the Rust v/s C++ post. It's been hours I have been hacking through, reading everything I can find about Rust in ML/DL. It won't be hard to say that most of them are abandoned. Some are still on but it seems like, there's no way they are comaprable to C++ alternatives. Without much of a community around, it's been pretty hard to get through anything or maybe I haven't searched enough. I also had a discussion with my friends who are into ML & they suggested me to go for C++. Let's come to what are the current examples in Rust. The best one I found was [**huggingface tokenizers**](https://github.com/huggingface/tokenizers). There were also crates like [`tch-rs`](https://github.com/LaurentMazare/tch-rs) which provide Rust bindings for C++ torch api. 

So, what is the state of rust in ML/DL? It's progressing slowly and steadily, but way further from maturity. It's very young and I guess someday we will be able to use Rust for ML/DL, but as of now I think am going to give it a pass. I know it feels like giving up, trust me even I felt so, but the fact is I will be working a full time job as a <ins>*SDE at TCS*</ins> soon. With a full time job, I need to be more cautious with how I use my time or spend my energies. While it's a lovely idea to explore my fantasies but I guess I should be productive at the same time. Also, I will be working with java alot at TCS so that would be like one more langugae added to the mix. In the end, there's no effective learning. Lately, by thinking through things I have learnt that I tend to go for new things and fancy stuff. But actually that's not what's important. Whatever you do,  you need **rock solid** fundamentals. Tooling is trivial as long as you really know what you're doing. So, I will be giving Rust a pass for now. But definitely one day I will come back to visit rust once again. Till then rather than learning a new language, I will stick what I have atleast some experience with, .i.e. **Python, C++ or Java** 

As always, thank you for reading 😊😃!



