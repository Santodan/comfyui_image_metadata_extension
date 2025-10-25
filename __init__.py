from .modules.nodes.node import SaveImageWithMetaData, CreateExtraMetaData

# (base_name, class_ref, display_name)
node_definitions = [
    ("SaveImageWithMetaData", SaveImageWithMetaData, "Save Image With MetaData"),
    ("CreateExtraMetaData", CreateExtraMetaData, "Create Extra MetaData"),
]

NODE_CLASS_MAPPINGS = {
    f"{base_name}": class_ref for base_name, class_ref, _ in node_definitions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"{base_name}": f"{display_name}" for base_name, _, display_name in node_definitions
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
