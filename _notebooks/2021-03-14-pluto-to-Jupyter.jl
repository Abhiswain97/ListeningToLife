### A Pluto.jl notebook ###
# v0.11.10

using Markdown
using InteractiveUtils

# ╔═╡ e3311240-8431-11eb-1d8d-61736f938fc4
md"""

# Pluto.jl + fastpages!

> converting pluto.jl notebook to jupyter and cleaning them to host as a fastpages blog post

- toc: true
- categories: [blog post, fastai, fastpages]
- comments: true
"""

# ╔═╡ 17224d40-8431-11eb-07db-d5e5f18760cb
md"""
Hi all! Today's post is for people who want to use Julia to write blog posts. I expect that you must have heard about this cool thing by [fast.ai](https://github.com/fastai) folks called [fastpages](https://fastpages.fast.ai/)

# What is fastpages and why to use it?

> An easy to use blogging platform, with support for Jupyter notebooks, Word docs, and Markdown.

Whole of my blog is made using fastpages. It takes just a few minutes to setup a whole blog using fastpages. Now, the best feature of fastpages is that you can use Jupyter notebooks to write a post and fastpages will automatically convert them into a blog post on your blog. Now this is seriously cool! Not just that you can write posts using regular markdown files, and also word document. fastpages will automatically render them as posts. You can refer their repo for more details: [fastpages](https://github.com/fastai/fastpages).

# Julia + fastpages

I have been trying julia out for few days now and instead of Python I wanted to use Julia to write blog posts. You can install the IJulia package and use Jupyter notebooks to write blog posts just like you do it in python. However, I would like to bring to your notice another very nice tool which I found about during JuliaCon 2020. I have used it a lot now and I can tell you that it addresses a lot of pitfalls of Jupyter Notebooks. It kind of changes the game. Currently, it only supports Julia and not Python. It's called [Pluto.jl](https://github.com/fonsp/Pluto.jl).

# Pluto.jl

Pluto.jl is: 

> Simple reactive notebooks for Julia

Let me explain it to you what **"reactivity"** is. In Jupyter notebook, say you have used a function which you haven't imported. Naturally, as you run the cell it would raise an error. To fix this error, you go all the way to the top where your imports are, import the particular function and re-run all the cells where the function is used. This is where Pluto.jl shines over Jupyter notebook. In Pluto as you import the function all the cells which depend on it are re-run automatically. It's pretty smart in figuring out dependencies. This is a very powerful feature.

# Pluto.jl + fastpages!

Finally we look at how we can use Pluto.jl to write blog posts. Now, fastpages requires the notebooks to be Ipython notebooks,to be able to render them as posts. So, what do we do? We will convert the Pluto.jl notebooks to Ipython notebook first and then use them. 

To do this, there's this tool made by the creator of Pluto.jl we can use -> [pluto-jl-jupyter-conversion](https://observablehq.com/@olivier_plas/pluto-jl-jupyter-conversion). Pluto saves your notebooks as .jl files. All you have to do is upload your Pluto.jl notebook and it will convert it into Jupyter notebook after which u can download it.

There's only one issue. The converted notebook has all the cells as code cells. Even the cells we used as markdown cells are converted to code cells. We need to fix this!
I have made a simple utility to solve this issue -> [pluto-jl-jupyter-conversion-cleaner](https://github.com/Abhiswain97/pluto-jl-jupyter-conversion-cleaner)

So, finally these are the steps you need to follow to convert your Pluto.jl notebook into a Jupyter notebook you can use with fastpages 

## Steps

- Use [pluto-jl-jupyter-conversion](https://observablehq.com/@olivier_plas/pluto-jl-jupyter-conversion) to convert the Pluto.jl notebook to a Ipython notebook. Download the Ipython notebook.

- After that:

```
git clone https://github.com/Abhiswain97/pluto-jl-jupyter-conversion-cleaner.git
cd pluto-jl-jupyter-conversion-cleaner
python cleaner.py --fname <your converted .ipynb file>
```

The converted file is saved as file-name-cleaned.ipynb in the same folder as your original Jupyter notebook.

Now, all you got to do is make it in the format fastpages expects, it has to be saved as: **yyyy-mm-dd-name.ipynb**. There are various other options you can specify. Have a look at their docs at: [fastpages](https://fastpages.fast.ai/)

Yay! we finally have a Jupyter notebook we can use with fastpages!

If you made it this far, then let me tell you something, the post you're reading right now is written using Pluto.jl & fastpages! 
"""

# ╔═╡ Cell order:
# ╠═e3311240-8431-11eb-1d8d-61736f938fc4
# ╠═17224d40-8431-11eb-07db-d5e5f18760cb
