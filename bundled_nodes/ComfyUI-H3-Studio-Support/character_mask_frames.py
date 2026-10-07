class H3CharacterSAMCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required': {'checkpoint_name': (folder_paths.get_filename_list('checkpoints'), {
            'tooltip': 'Shared SAM3 model choice for its loader and both mask-cache identities.'})}}

    # Comfy's legacy checkpoint dropdowns use a list of choices as their input
    # type. '*' is the supported bridge for one validated dropdown feeding all
    # three dropdown inputs without forcing the model loader on a cache hit.
    RETURN_TYPES = ('*',)
    RETURN_NAMES = ('checkpoint_name',)
    FUNCTION = 'select'
    CATEGORY = 'MiniMax Safe/Character'

    @classmethod
    def IS_CHANGED(cls, checkpoint_name):
        import os
        import folder_paths
        path = folder_paths.get_full_path('checkpoints', checkpoint_name)
        if not path:
            return ('missing', checkpoint_name)
        stat = os.stat(path)
        return (stat.st_size, stat.st_mtime_ns)

    def select(self, checkpoint_name):
        import folder_paths
        if checkpoint_name not in folder_paths.get_filename_list('checkpoints'):
            raise ValueError('Choose an installed SAM3 checkpoint.')
        return (checkpoint_name,)
