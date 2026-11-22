"""Decide whether fetching a job-posting URL may proceed.

Adapted from ``tools/robots_check.py`` in the MIT-licensed ai-job-search
repository (Copyright (c) 2026 Mads Lorentzen); see NOTICE. The gate is
enforced by ``agent/url_fetch.py`` on every hop; the robots.txt request and
the page request both identify honestly as ``ResumeOptimizer/1.0`` - no
browser impersonation.

RFC 9309 walk, deliberately on the cautious side:
  * longest-match wins; on equal specificity Disallow wins
  * a Disallow for either "*" or "ResumeOptimizer" blocks the fetch
  * blank lines inside a record do not end it (Python's robotparser drops
    rules in that case, which fails open)
  * one leading UTF-8 BOM is skipped (left in, it hides a first
    "User-agent" line and drops that whole group, which fails open)
  * 404 means no published policy, which is permission
  * any other failure to read robots.txt leaves permission unconfirmed,
    and the fetch is refused - including a U+FFFD byte in a rule (a byte
    that is not valid UTF-8)
  * only http/https URLs with a host are checked; anything else is
    unconfirmed without a request

``gate(url)`` returns ``(code, msg)``: code 0 = fetch may proceed, 1 = do
not (``msg`` says why). Used by ``agent/url_fetch.py``; see its
``_fetch_job_description``.
"""
import re
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

USER_AGENT = "ResumeOptimizer/1.0"


def is_robots_body(text):
    """Does this actually look like a robots.txt?

    A misconfigured host can answer /robots.txt with 200 and an HTML error page.
    That body parses to zero rules, and zero rules read as "allowed" - so a
    soft-200 granted permission that was never given. An empty or whitespace-only
    body IS a valid allow-all under RFC 9309 and stays allowed; a non-empty body
    with no recognised directive is treated as unreadable.
    """
    text = text.removeprefix('\ufeff')          # byte-order mark, see _groups
    if not text.strip():
        return True
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip().lower()
        if ':' in line and line.split(':', 1)[0].strip() in (
                'user-agent', 'allow', 'disallow', 'sitemap', 'crawl-delay', 'host',
        ):
            return True
    return False


def has_undecodable_rule(text):
    """Does a User-agent, Allow or Disallow line hold U+FFFD?

    U+FFFD stands in for a byte that did not decode as UTF-8. A non-conformant
    robots.txt saved in a legacy code page (cp1254, ISO-8859-9) with raw
    non-ASCII bytes in a rule yields a pattern that can never match: the rule
    was silently skipped and the path read as allowed. Such a body is
    unreadable, not permissive. The field is named with U+FFFD removed, so a
    stray byte before "Disallow" cannot hide the line. U+FFFD in a comment or
    in any other field (Sitemap, Crawl-delay, ...) decides nothing and is
    ignored.
    """
    for raw in text.removeprefix('\ufeff').splitlines():
        line = raw.split('#', 1)[0]
        if '\ufffd' in line and ':' in line:
            field = line.split(':', 1)[0].replace('\ufffd', '').strip().lower()
            if field in ('user-agent', 'allow', 'disallow'):
                return True
    return False


def _groups(text):
    """user-agent -> [(is_allow, pattern)], tolerating blank lines inside a record.

    One leading U+FEFF is skipped, as Google's reference parser does. A file
    saved with a UTF-8 byte-order mark otherwise hid its first field: a leading
    "User-agent: *" went unseen, every rule in its group was dropped for want
    of an agent, and "Disallow: /" read as allow-all.
    """
    out, agents, expect = {}, [], True
    for raw in text.removeprefix('\ufeff').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line or ':' not in line:
            continue
        field, _, value = line.partition(':')
        field, value = field.strip().lower(), value.strip()
        if field == 'user-agent':
            if not expect:
                agents, expect = [], True
            agents.append(value.lower())
            out.setdefault(value.lower(), [])
        elif field in ('allow', 'disallow') and agents:
            expect = False
            for a in agents:
                out[a].append((field == 'allow', value))
    return out


def _match(pattern, path):
    """RFC 9309 wildcard match; returns match length or -1.

    The pattern is percent-decoded to match the already-decoded path. Without
    this, "Disallow: /foo%20bar" never matched "/foo bar" and the rule was
    silently skipped - a fail-open on any site that encodes its own rules.
    Under RFC 9309 section 2.2.2, '$' anchors the end of the URL only when
    at the end of the pattern; mid-pattern '$' is matched literally.

    Matching is a table walk in O(len(pattern) * len(path)), never a regex.
    Each '*' used to become '.*', and a non-matching rule with repeated
    wildcard segments ("/*a*a*a...*b$") sent Python's backtracking engine into
    exponential time - minutes at 16 segments - on a file the site controls
    (GHSA-xhqf-8q8x-4jwm). Consecutive '*' match as one, as RFC 9309 implies.
    """
    if pattern == '':
        return -1
    pattern = unquote(pattern)
    end_anchor = pattern.endswith('$')
    pat = re.sub(r'\*{2,}', '*', pattern[:-1] if end_anchor else pattern)
    # reach[j] is True when the first j pattern characters match the path
    # consumed so far. A '*' extends a match over any character (reach[j]
    # carries forward) and also matches nothing (reach[j-1] on the same row).
    reach = [True] + [False] * len(pat)
    for j, c in enumerate(pat, 1):
        if c != '*':
            break
        reach[j] = True
    if reach[-1] and not end_anchor:
        return len(pattern)
    for ch in path:
        step = [False] * (len(pat) + 1)
        for j, c in enumerate(pat, 1):
            if c == '*':
                step[j] = step[j - 1] or reach[j]
            elif c == ch:
                step[j] = reach[j - 1]
        reach = step
        if reach[-1] and not end_anchor:
            return len(pattern)      # a prefix match is enough without '$'
    return len(pattern) if reach[-1] else -1


def allowed(text, agent, path):
    g = _groups(text)
    rules = g.get(agent.lower()) or g.get('*') or []
    best_len, best_allow = -1, True
    for is_allow, pat in rules:
        n = _match(pat, path)
        if n > best_len or (n == best_len and n >= 0 and not is_allow):
            best_len, best_allow = n, is_allow      # ties -> Disallow wins (cautious)
    return True if best_len < 0 else best_allow


def _fetch(url, ua):
    """urllib, not external tooling: the body is UTF-8 (RFC 9309) and a hard
    ceiling is set with timeout=12."""
    req = Request(url, headers={"User-Agent": ua, "Accept": "text/plain,*/*"})
    try:
        with urlopen(req, timeout=12) as res:
            return res.read().decode("utf-8", errors="replace"), res.status
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return body, exc.code
    # a failed read / timeout / connection issue -> unknown policy
    except (URLError, OSError, ValueError):
        raise


def gate(url):
    parts = urlsplit(url)  # urlsplit lower-cases the scheme
    # Only http(s) with a host is a page this tool can have a policy for; any
    # other input fails closed before a request is asked.
    if parts.scheme not in ('http', 'https'):
        return 1, f'UNCONFIRMED (unsupported URL scheme {parts.scheme!r}) - robots.txt could not be read'
    if not parts.netloc:
        return 1, 'UNCONFIRMED (no host in the URL) - robots.txt could not be read'
    path = unquote(parts.path) or '/'
    if parts.query:
        path += '?' + unquote(parts.query)
    robots = f'{parts.scheme}://{parts.netloc}/robots.txt'
    try:
        text, code = _fetch(robots, USER_AGENT)
    except Exception as e:
        return 1, f'UNCONFIRMED ({type(e).__name__}) - robots.txt could not be read'
    if code == 404:
        return 0, 'ALLOWED - no robots.txt published'
    if code != 200:
        return 1, f'UNCONFIRMED (HTTP {code}) - robots.txt could not be read'
    if not is_robots_body(text):
        return 1, ('UNCONFIRMED (HTTP 200 but the body is not a robots.txt) '
                   '- robots.txt could not be read')
    if has_undecodable_rule(text):
        return 1, ('UNCONFIRMED (HTTP 200 but a User-agent/Allow/Disallow '
                   'line is not valid UTF-8) - robots.txt could not be read')
    for agent in ('ResumeOptimizer', '*'):
        if not allowed(text, agent, path):
            return 1, (f'DISALLOWED for {agent} - robots.txt does not '
                       'permit this path')
    return 0, 'ALLOWED - robots.txt permits this path'
