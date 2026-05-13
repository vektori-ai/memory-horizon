path = "/usr/local/lib/python3.10/dist-packages/verl/workers/actor/dp_actor.py"
c = open(path).read()
assert "self.actor_optimizer.step()" in c, "pattern not found — check verl version"
c = c.replace(
    "self.actor_optimizer.step()",
    "torch.cuda.empty_cache(); self.actor_optimizer.step()",
)
open(path, "w").write(c)
print("verl dp_actor patch OK — empty_cache injected before optimizer step")
