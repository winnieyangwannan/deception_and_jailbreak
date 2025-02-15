import ray

ray.init()


@ray.remote
def hello():
    return "Hello from Ray!"


def multiple():
    ray.init(address="auto")
    print(ray.nodes())


print(ray.get(hello.remote()))
