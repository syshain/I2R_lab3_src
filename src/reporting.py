"""All console output, in one grammar.

    .. message       process noise: loading, capturing, iterating. Ignore it.
    == TITLE ==      a stage or section.
    -- Heading       a block of results within a stage.
      label   value  a result. Anything in this shape is a number to read off.
"""


def banner(title):
    """Print a section title between double rules."""
    print("=" * 64)
    print(" " + title)
    print("=" * 64)


def heading(title):
    """Print a sub-block title."""
    print("  -- %s" % title)


def progress(message):
    """Print process noise, visually set apart from results."""
    print("  .. %s" % message)


def field(label, value, note=''):
    """Print one result line: aligned label, formatted value, optional note."""
    text = "    %-11s %s" % (label, value)
    if note:
        text += "   %s" % note
    print(text)
