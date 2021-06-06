# https://github.com/flairNLP/flair/blob/master/flair/visual/ner_html.py
import html
from typing import Union, List

TAGGED_ENTITY = """
<mark class="entity" style="background: {color}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 3; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    {entity}
    <span style="font-size: 0.8em; font-weight: bold; line-height: 3; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">{label}</span>
</mark>
"""

PARAGRAPH = """<p>{sentence}</p>"""

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>{title}</title>
    </head>
    <body style="font-size: 16px; font-family: 'Segoe UI'; padding: 4rem 2rem">{text}</body>
</html>
"""


def split_to_spans(text, results):
    last_idx = 0
    spans = []
    for ent in results:
        if last_idx != ent["start"]:
            spans.append((text[last_idx : ent["start"]], None))
        spans.append((ent["text"], ent["entity"]))
        last_idx = ent["end"]
    if last_idx < len(text) - 1:
        spans.append((text[last_idx : len(text)], None))
    return spans

#https://www.computerhope.com/htmcolor.htm
def render_ner_html(
    text, 
    results,
    title="NER",
    colors={
        "PER": "#4CC417",
        "PERSON": "#4CC417",
        "ORG": "#2B65EC",
        "LOC": "#F88017",
        "MISC": "#4647EB",
        "GPE": "#F88017",
        "DATE": "#2B547E",
        "CARDINAL": "#736F6E",
        "NORP": "#FFE87C",
        "MONEY": "#736F6E",
        "PERCENT": "#CD7F32",
        "WORK_OF_ART": "#571B7E",
        "ORDINAL": "#736F6E",
        "EVENT": "#FFE87C",
        "TIME": "#FFE5B4",
        "FAC": "#571B7E",
        "QUANTITY": "#736F6E",
        "PRODUCT": "#571B7E",
        "LANGUAGE": "#FFE87C",
        "LAW": "#2B65EC",
        "O": "#ddd",
    },
    default_color: str = "#ddd",
    wrap_page=True,
) -> str:
    """
    :param sentences: single sentence or list of sentences to convert to HTML
    :param title: title of the HTML page
    :param colors: dict where keys are tags and values are color HTML codes
    :param default_color: color to use if colors parameter is missing a tag
    :param wrap_page: if True method returns result of processing sentences wrapped by &lt;html&gt; and &lt;body&gt; tags, otherwise - without these tags
    :return: HTML as a string
    """
    spans = split_to_spans(text, results)
    spans_html = list()
    for fragment, tag in spans:
        escaped_fragment = html.escape(fragment).replace("\n", "<br/>")
        if tag:
            escaped_fragment = TAGGED_ENTITY.format(
                entity=escaped_fragment,
                label=tag,
                color=colors.get(tag, default_color),
            )
        spans_html.append(escaped_fragment)
    final_text = PARAGRAPH.format(sentence="".join(spans_html))

    if wrap_page:
        return HTML_PAGE.format(text=final_text, title=title)
    else:
        return final_text