import yaml


def is_empty_val(value):
    return not value


class YamlDump(yaml.SafeDumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(YamlDump, self).increase_indent(flow, False)
