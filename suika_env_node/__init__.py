from gymnasium.envs.registration import register as register

register(
    id="SuikaEnvNode-v0",
    entry_point="suika_env_node.suika_node_env:SuikaNodeEnv",
)
