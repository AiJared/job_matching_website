from django import template

register = template.Library()

@register.filter
def split_string(value, separator=','):
    """Split a string by separator and return a list"""
    if value:
        return [item.strip() for item in value.split(separator)]
    return []