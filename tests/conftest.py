from hypothesis import settings

# Keep runs reproducible and leave no example database on disk.
settings.register_profile("repo", database=None, derandomize=True)
settings.load_profile("repo")
