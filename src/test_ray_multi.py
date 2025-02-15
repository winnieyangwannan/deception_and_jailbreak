import ray

ray.init()


@ray.remote
def multiple():
    ray.shutdown()
    ray.init(address="auto")
    print(ray.nodes())


print(ray.get(multiple.remote()))
