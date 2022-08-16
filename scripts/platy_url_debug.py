#%%
from pymaid.client import CatmaidInstance
from pymaid.utils import _make_iterable
from pymaid.cache import Cache

#%%
url = "https://catmaid.jekelylab.ex.ac.uk/#"
names = "PRCal1"

#%%
# get skids by name function stuff
names = list(set(_make_iterable(names, force_type=str)))
print(names)

#%%
post = []
for n in names:
    post.append({"name": n, "with_annotations": False, "name_exact": True})
post[-1]["name_exact"] = False
print(post)


#%%
for n in post:
    print(n)


#%%
print(isinstance(url, str))
print(_make_iterable(url))
post = None
post = [post] * len(url) if isinstance(post, (type(None), dict, bool)) else post
print(post)
print(len(url))
for u, p in zip(url, post):
    print(u)
    print(p)

c = Cache()
tup = (url, str(post))
print(tup)
print(c.__getitem__(tup))

# %%
"""
url = rm.server
print(url)
# following the code in the make_url function
for arg in [rm.project_id, "annotations", "query-targets"]:
    arg_str = str(arg)
    joiner = "" if url.endswith("/") else "/"
    relative = arg_str[1:] if arg_str.startswith("/") else arg_str
    url = requests.compat.urljoin(url + joiner, relative)
    print(url)
"""
