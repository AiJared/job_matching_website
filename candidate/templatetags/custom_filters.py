from django import template

register = template.Library()

@register.filter
def split_string(value, separator=','):
    """
    Split a string by comma/newline and return a list of cleaned skills
    """
    if not value:
        return []
    
    # Normalize both commas and newlines to a single separator
    normalized = value.replace('\n', ',').replace(';', ',')
    return [item.strip() for item in normalized.split(',') if item.strip()]
