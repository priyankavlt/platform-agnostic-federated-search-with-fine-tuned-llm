from fedsearch.ingest.entities import EntityRegistry


def test_resolve_aliases_fuzzy_and_ids():
    r = EntityRegistry()
    r.add("customer", "CUST-0001", "Globex", "globex.com", display="Globex")
    r.add("product", "sso", "Single Sign-On", "SSO")
    assert r.resolve("customer", "globex") == "customer:CUST-0001"
    assert r.resolve("customer", "Glboex") == "customer:CUST-0001"  # typo
    assert r.resolve("product", "sso") == "product:sso"
    assert r.resolve("ticket", "tck-1042") == "ticket:TCK-1042"
    assert r.resolve("customer", "Initech") is None


def test_extract_links_email_domain_and_ids():
    r = EntityRegistry()
    r.add("customer", "CUST-0001", "Globex", "globex.com")
    found = r.extract("Ticket TCK-1001 from ops@globex.com shows E-SSO-112")
    assert {"ticket:TCK-1001", "error_code:E-SSO-112", "customer:CUST-0001"} <= found
