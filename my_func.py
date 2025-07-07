
def cprint(f="",s="",t="", **kwargs):
    # colorize second arguments
    # cprint(1,2,3)
    # =>  123  (only 2 is red)
    # cprint([],"a","b",color="blue")
    # => ab   (only a is blue)
# 赤	31	\033[31m
# 緑	32	\033[32m
# 黄	33	\033[33m
# 青	34	\033[34m
  match kwargs.get("color"):
    case "red":
      print(f"{f}\033[1;31m{s}\033[0m{t}")
    case "green":
      print(f"{f}\033[1;32m{s}\033[0m{t}")
    case "yellow":
      print(f"{f}\033[1;33m{s}\033[0m{t}")
    case "blue":
      print(f"{f}\033[1;34m{s}\033[0m{t}")
    case _:
      print(f"{f}\033[1;31m{s}\033[0m{t}")