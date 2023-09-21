import ray
import abc


@ray.remote
class ActorX(abc.ABC):
    def __init__(self, actor_config) -> None:
        super().__init__()
        # ds.prepare()
        # self._actor_model = nn.Module.load_from()
        # 8卡


    def is_ready(self) -> bool:
        return True

    def train(self):
        self._actor_model.train()

    def forward(self, inputs):
        return self._actor_model(inputs)

#  1 ActorGroup  <-----> wrapper proxy 
#    - 8 remote_actors <---> 进程 <--->
#       remote_actor NODE0 0
#       remote_actor NODE0 1
#

# a.f.remote()
