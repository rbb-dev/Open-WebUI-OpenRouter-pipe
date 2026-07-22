import json

import pytest

from open_webui_openrouter_pipe.tools.citation_harvester import (
    BUILTIN_CITATION_TOOLS,
    harvest_tool_citations,
)

EXA_SEARCH_FIXTURE = "Title: Consumer Price Index, Australia, May 2026\nURL: https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/latest-release\nPublished: 2026-06-24T00:00:00.000Z\nAuthor: N/A\nHighlights:\nConsumer Price Index, Australia, May 2026 | Australian Bureau of Statistics\n...\nConsumer Price Index, Australia\n...\nThe Consumer Price Index (CPI) measures household inflation and includes statistics about price change for categories of household expenditure.\n...\nMay 2026\n...\n24/06/2026\n...\n24/06/2026 11:30am AEST\n...\n## Key statistics\n...\nIn the 12 months to May 2026:\n...\n- The Consumer Price Index (CPI) rose 4.0%, down from 4.2% in the 12 months to April 2026.\n- The largest contributors to annual inflation were Housing (+6.5%), Food and non-alcoholic beverages (+3.3%) and Transport (+3.3%).\n- Trimmed mean inflation was 3.6%, up from 3.4% in the 12 months to April 2026.\n...\nIn the month of May, the CPI fell 0.7% in original terms and 0.1% in seasonally adjusted terms.\n...\n| May-26\n...\n### CPI annual inflation fell, while Trimmed mean inflation rose\n...\nCPI annual inflation was 4.0 per cent in the 12 months to May 2026, down from 4.2 per cent in the 12 months to April 2026.\n...\nTrimmed mean inflation was 3.6 per cent in the 12 months to May 2026, up from 3.4 per cent in the 12 months to April 2026.\n...\n12 months to May 2026\n\n---\n\nTitle: CPI rose 4.0% in the year to May 2026\nURL: https://www.abs.gov.au/media-centre/media-releases/cpi-rose-40-year-may-2026\nPublished: 2026-06-24T00:00:00.000Z\nAuthor: N/A\nHighlights:\nCPI rose 4.0% in the year to May 2026 | Australian Bureau of Statistics\n...\n# CPI rose 4.0% in the year to May 2026\n...\nThe Consumer Price Index (CPI) rose 4.0 per cent in the 12 months to May 2026, according to the latest data from the Australian Bureau of Statistics (ABS).\n...\nRachael McCririck, ABS head of prices statistics, said: ‘Annual CPI inflation in May was 4.0 per cent, down from 4.2 per cent in the year to April.’\n...\nThe largest contributor to annual inflation in May was Housing, which rose by 6.5 per cent. This was followed by a 3.3 per cent rise in Food and non-alcoholic beverages and a 3.3 per cent rise in Transport.\n...\n| | Change\n...\nprevious month (%) | Annual change (%) |\n...\n| --- | --- | --- |\n|\n...\n0.2 | |\n...\n| Jun-2\n...\n0.4 | |\n...\n| Jul-24 | 0.3 | |\n...\n| Aug-24 | -0.3 | |\n...\n| Sep-24 | 0.1 | |\n| Oct-24 | -0.2 | |\n| Nov-24 | 0.4 | |\n| Dec-24 | 0.7 | |\n| Jan-25 | 0.3 | |\n| Feb-25 | 0.1 | |\n| Mar-25 | 0.3 | |\n| Apr-25 | 0.7 | 2.4 |\n| May-25 | -0.5 | 2.1 |\n| Jun-25 | 0.1 | 1.9 |\n| Jul-25 | 1.3 | 3.0 |\n| Aug-25 | -0.1 | 3.2 |\n| Sep-25 | 0.5 | 3.6 |\n| Oct-25 | 0.0 | 3.8 |\n| Nov-25 | 0.0 | 3.4 |\n| Dec-25 | 1.0 | 3.8 |\n| Jan-26 | 0.4 | 3.8 |\n| Feb-26 | 0.0 | 3.7 |\n| Mar-26 | 1.1 | 4.6 |\n| Apr-26 | 0.4 | 4.2 |\n| May-26 | -0.7 | 4.0 |\n...\n‘Trimmed mean annual inflation was 3.6 per cent in the 12 months to May 2026, up from 3.4 per cent in the 12 months to April 2026,’ Ms McCririck said.\n...\nAnnual Housing inflation was 6.5 per cent in the 12 months to May. This reflects rising costs for Electricity, New dwellings and Rents.\n...\nAnnual inflation for Food and non-alcoholic beverages was 3.3 per cent, up from 2.8 per cent in April. Food inflation was driven by higher prices for Meals out and takeaway, which rose by 4.0 per cent in the 12 months to May 2026.\n...\n‘Price growth for Transport eased from what we saw in April, rising 3.3 per cent in annual terms, down from a 6.6 per cent rise in the 12 months to April 2026.\n\n---\n\nTitle: Consumer Price Index, Australia | Australian Bureau of Statistics\nURL: https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/sort\nPublished: N/A\nAuthor: N/A\nHighlights:\nConsumer Price Index, Australia | Australian Bureau of Statistics\n...\nRelease date and time 24 June 2026 Release date and time: 24/06/2026In the 12 months to May 2026:The Consumer Price Index (CPI) rose 4.0%, down from 4.2% in the 12 months to April 2026.The largest contributors to annual inflation were Housing (+6.5%), Food and non-alcoholic beverages (+3.3%) and Transport (+3.3%).Trimmed mean inflation was 3.6%, up from 3.4% in the 12 months to April 2026.In the month of May, the CPI fell 0.7% in original terms and 0.1% in seasonally adjusted terms.Reference periodMay 2026\n\n---\n\nTitle: Consumer Price Index, Australia\nURL: https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia\nPublished: N/A\nAuthor: N/A\nHighlights:\nConsumer Price Index, Australia | Australian Bureau of Statistics\n...\nConsumer Price Index, Australia, May 2026 Latest release\n\n---\n\nTitle: Consumer Price Index, Australia methodology, May 2026 | Australian Bureau of Statistics\nURL: https://www.abs.gov.au/methodologies/consumer-price-index-australia-methodology/may-2026\nPublished: N/A\nAuthor: N/A\nHighlights:\nConsumer Price Index, Australia methodology, May 2026 | Australian Bureau of Statistics\n...\n# Consumer Price Index, Australia methodology\n...\nMay 2026\n...\n24/06/2026\n...\n- Consumer Price Index, Australia methodology Reference Period\n...\n2026\n...\nConsumer Price Index\n...\n2026\n...\nConsumer Price Index\n...\n2026\n...\nRelease date and time\n...\n24/06/2026 11:30am AEST\n...\nThe CPI is a general measure of price change for goods and services purchased by Australian households.\n...\n## Data release\n...\n- The Consumer Price Index is released monthly, on the last Wednesday of each month (for the previous month's reference period), except the December publication, which is released in early January. In most cases, the CPI is released four weeks after the end of the reference month.\n...\n- Quarterly tables will be included in every monthly\n...\ngoing forward. New quarterly data will be available in every third monthly publication (i.e., March, June\n...\n- Monthly and quarterly CPI data have an index reference period of September 2025 = 100.00. The quarterly data was re-referenced for the December 2025 quarter. Prior to this, the index reference period was 2011-12 = 100.0.\n- Release dates are published under the Future releases section of the publication and in the ABS Release Calendar.\n\n---\n\nTitle: Headline inflation eases, but underlying inflation highest since 2024 - ABC News\nURL: https://www.abc.net.au/news/2026-06-24/australia-cpi-inflation-may-2026-bureau-of-statistics/106835946\nPublished: 2026-06-24T01:51:46.000Z\nAuthor: N/A\nHighlights:\nHeadline inflation eased in May, but underlying inflation strengthened again.\n...\nTrimmed mean inflation, the Reserve Bank's preferred measure of underlying inflation, increased from an annual pace of 3.4 per cent to 3.6 per cent.\n...\nHeadline inflation eased in May, with consumer prices increasing at an annual pace of 4 per cent, down from 4.2 per cent in April.\n...\nBut trimmed mean inflation, the Reserve Bank's preferred measure of underlying inflation, still increased to 3.6 per cent in May, up from 3.4 per cent.\n...\nAnd the Bureau of Statistics says underlying inflation is now running at its highest annual pace since the September quarter of 2024.\n\n---\n\nTitle: Inflation lying in wait to strike another rate blow\nURL: https://aapnews.aap.com.au/news/inflation-tipped-to-rise-despite-falling-fuel-prices\nPublished: 2026-06-24T05:16:46.000Z\nAuthor: N/A\nHighlights:\nA mixed consumer price report, released by the Australian Bureau of Statistics on Wednesday, was softer and stronger than forecasters had predicted.\n...\nThe core inflation measure, which strips out volatile items and is more closely watched by the RBA, edged up to 3.6 per cent in May, from 3.4 per cent a month earlier.\n...\nHeadline inflation fell to four per cent annually, from 4.2 per cent in April.\n...\n9 per cent\n...\nCPI rose 4.0% in the year to May 2026Find out more https://t.co/Vq5I5szNw1\n...\n— Australian Bureau of Statistics (@ABSStats) June 24, 2026\n...\nBut ANZ Bank economists Madeline Dunk and Adam Boyton said the report suggested the trimmed mean was tracking to 3.7 per cent for the June quarter, which would be a touch lower than the RBA's forecast of 3.8 per cent.\n\n---\n\nTitle: Australia CPI Inflation Drops to 4.0% in May 2026, Housing Costs Lead\nURL: https://www.indexbox.io/blog/australia-cpi-inflation-drops-to-40-in-may-2026-housing-costs-lead/\nPublished: 2026-06-24T08:58:54.000Z\nAuthor: N/A\nHighlights:\nAustralia CPI Inflation May 2026: Annual Rate Falls to 4.0%, Housing Up 6.5% - News and Statistics - IndexBox\n...\nAnnual inflation falls to 4.0% in May 2026\n...\nat 6.5% annual rise\n...\n# Australia CPI Inflation Drops to 4.0% in May 2026, Housing Costs Lead\n...\nThe Australian Bureau of Statistics (ABS) has released the latest Consumer Price Index(CPI) data, which measures household inflation by tracking price changes across various expenditure categories.\n...\nAccording to the ABS, annual CPI inflation stood at 4.0 percent in the 12 months to May 2026, a decline from the 4.2 percent recorded in the 12 months to April 2026. The largest contributors to this annual inflation were Housing, which rose 6.5 percent, followed by Food and non-alcoholic beverages and Transport, each increasing 3.3 percent.\n...\nThe trimmed mean inflation, a measure of underlying inflation, was 3.6 percent for the 12 months to May 2026, up from 3.4 percent in the previous period. On a monthly basis, the CPI fell 0.7 percent in original terms and 0.1 percent in seasonally adjusted terms for May.\n...\nAnnual goods inflation was 4.2 percent in the year to May 2026, down from 4.7 percent in the year to April 2026. Key contributors included Electricity, which surged 21.1 percent, and New dwellings, which rose 5.6 percent. Automotive fuel increased 7.7 percent over the same period, a significant slowdown from the 18.6 percent rise recorded in the 12 months to April 2026.\n...\nThe Housing group saw a 6.5 percent annual increase, accelerating from 6.3 percent in the 12 months to April 2026. Electricity costs rose 21.1 percent, primarily due to the conclusion of Commonwealth and State Government electricity rebates. Excluding the impact of those rebates, electricity prices rose 3.9 percent over the year. New dwelling prices increased 5.6 percent, driven by project home builders raising base prices to cover higher labour and material costs. Rental prices rose 3.6 percent, reflecting sustained low vacancy rates in most capital cities.\n...\nthat from February 202\n...\n, the CPI\n...\nbe published on the fourth Wednesday of each month instead of the final Wednesday, following a review that determined an earlier release would not affect data quality. Additionally, a mid-2026 weight update was considered but deemed unnecessary after monitoring household spending patterns; the CPI weights will next be updated for January\n...\n2027, scheduled for release on 24 February 2027.\n\n---\n\nTitle: CPI rose 4.0% in the year to May 2026 | Australian Bureau of Statistics\nURL: https://www.linkedin.com/posts/absstats_cpi-rose-40-in-the-year-to-may-2026-activity-7475397310814826496-m9Yw\nPublished: 2026-06-24T00:00:00.000Z\nAuthor: Australian Bureau of Statistics\nHighlights:\n# CPI rose 4.0% in the year to May 2026 | Australian Bureau of Statistics · LinkedIn · 2026-06-24\n...\nCPI rose 4.0% in the year to May 2026\n\n---\n\nTitle: Release schedule | Australian Bureau of Statistics\nURL: https://www.abs.gov.au/about/key-priorities/big-data-timely-insights-phase-2/complete-monthly-measure-cpi/release-schedule\nPublished: N/A\nAuthor: N/A\nHighlights:\ncomplete Monthly CPI on\n...\nThe Monthly CPI will be published on the last Wednesday of the month, following the end of the reference month. The exception to this will be November, which will be released on the 7 January 2026 owing to the Christmas / New Year holiday period. The release will be published on the ABS website in a new publication titled Consumer Price Index, Australia and will be available on the Price indexes and inflation page. Table 1 outlines the release schedule for six months following the initial release.\n...\n| Table 1 Consumer Price Index future releases |\n| --- |\n| Reference Month | Release Date |\n| Consumer Price Index, Australia, November 2025 | 07/01/2026 |\n| Consumer Price Index, Australia, December 2025 | 28/01/2026 |\n| Consumer Price Index, Australia, January 2026 | 25/02/2026 |\n| Consumer Price Index, Australia, February 2026 | 25/03/2026 |\n| Consumer Price Index, Australia, March 2026 | 29/04/2026 |\n| Consumer Price Index, Australia, April 2026 | 27/05/2026 |\n...\nThe transition to the complete Monthly CPI represents a significant change to the ABS systems, data and business processes. The first six months of publication will be used to bed in the new systems, resolve any issues, and assess whether the release schedule can be brought forward without compromising statistical quality. The ABS will review the publication schedule by May 2026."

EXA_FETCH_FIXTURE = "# Climate change\nURL: https://www.who.int/news-room/fact-sheets/detail/climate-change-and-health\n\nClimate change Skip to main content\n\nWHO /A. Craggs\n\n© Credits\n\n# Climate change\n\n## Key facts\n\n- Climate change is directly contributing to humanitarian emergencies from heatwaves, wildfires, floods, tropical storms and hurricanes and they are increasing in scale, frequency and intensity.\n- Research shows that 3.6 billion people already live in areas highly susceptible to climate change. Between 2030 and 2050, climate change is expected to cause approximately 250 000 additional deaths per year, from undernutrition, malaria, diarrhoea and heat stress alone.\n- The direct damage costs to health (excluding costs in health-determining sectors such as agriculture and water and sanitation) is estimated to be between US$ 2–4 billion per year by 2030.\n- Areas with weak health infrastructure – mostly in developing countries – will be the least able to cope without assistance to prepare and respond.\n- Reducing emissions of greenhouse gases through better transport, food and energy use choices can result in very large gains for health, particularly through reduced air pollution.\n\n---\n\n## Overview\n\nClimate change presents a fundamental threat to human health. It affects the physical environment as well as all aspects of both natural and human systems – including social and economic conditions and the functioning of health systems. It is therefore a threat multiplier, undermining and potentially reversing decades of health progress. As climatic conditions change, more frequent and intensifying weather and climate events are observed, including storms, extreme heat, floods, droughts and wildfires. These weather and climate hazards affect health both directly and indirectly, increasing the risk of deaths, noncommunicable diseases, the emergence and spread of infectious diseases, and health emergencies.\n\nClimate change is also having an impact on our health workforce and infrastructure, reducing capacity to provide universal health coverage (UHC). More fundamentally, climate shocks and growing stresses such as changing temperature and precipitation patterns, drought, floods and rising sea levels degrade the environmental and social determinants of physical and mental health. All aspects of health are affected by climate change, from clean air, water and soil to food systems and livelihoods. Further delay in tackling climate change will increase health risks, undermine decades of improvements in global health, and contravene our collective commitments to ensure the human right to health for all.\n\n## Climate change impacts on health\n\nThe Intergovernmental Panel on Climate Change's (IPCC) Sixth Assessment Report (AR6) concluded that climate risks are appearing faster and will become more severe sooner than previously expected, and it will be harder to adapt with increased global heating.\n\nIt further reveals that 3.6 billion people already live in areas highly susceptible to climate change. Despite contributing minimally to global emissions, low-income countries and"


def test_builtin_set_contents():
    assert BUILTIN_CITATION_TOOLS == {
        "search_web", "fetch_url", "view_file", "view_knowledge_file", "query_knowledge_files",
    }


def test_exa_search_yields_unique_url_sources_with_titles():
    results = harvest_tool_citations(EXA_SEARCH_FIXTURE)
    assert len(results) == 10
    urls = [r[0] for r in results]
    assert len(set(urls)) == 10
    assert all(u.startswith("http") for u in urls)
    assert results[0][1] == "Consumer Price Index, Australia, May 2026"
    assert results[0][0] == "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/latest-release"


def test_exa_search_dashes_in_highlights_do_not_oversplit():
    assert EXA_SEARCH_FIXTURE.count("---") >= 10
    results = harvest_tool_citations(EXA_SEARCH_FIXTURE)
    assert len(results) == 10


def test_exa_search_highlights_body_url_not_harvested():
    poisoned = EXA_SEARCH_FIXTURE.replace(
        "Highlights:\n", "Highlights:\nURL: http://attacker.example/inline\n", 1,
    )
    results = harvest_tool_citations(poisoned)
    assert all("attacker.example" not in r[0] for r in results)


def test_exa_search_forged_envelope_requires_full_shape():
    forged = EXA_SEARCH_FIXTURE + "\n\n---\n\nTitle: Fake\nURL: http://attacker.example/forged\nno-envelope"
    results = harvest_tool_citations(forged)
    assert all(r[0] != "http://attacker.example/forged" for r in results)


def test_exa_fetch_header_url_single_source():
    results = harvest_tool_citations(EXA_FETCH_FIXTURE)
    assert len(results) == 1
    url, title, snippet = results[0]
    assert url == "https://www.who.int/news-room/fact-sheets/detail/climate-change-and-health"
    assert title == "Climate change"
    assert snippet


def test_exa_fetch_body_url_lines_not_harvested():
    poisoned = EXA_FETCH_FIXTURE + "\n\n## Injected\nURL: http://attacker.example/body\n"
    results = harvest_tool_citations(poisoned)
    assert len(results) == 1
    assert "attacker.example" not in results[0][0]


def test_json_results_array_url_title_yields_sources():
    payload = json.dumps({"results": [
        {"url": "https://a.example/one", "title": "One", "text": "alpha"},
        {"url": "https://a.example/two", "title": "Two", "snippet": "beta"},
    ]})
    results = harvest_tool_citations(payload)
    assert [(r[0], r[1]) for r in results] == [
        ("https://a.example/one", "One"),
        ("https://a.example/two", "Two"),
    ]
    assert results[0][2] == "alpha"


def test_json_link_and_href_keys_recognized():
    payload = json.dumps([
        {"link": "https://l.example/", "name": "L"},
        {"href": "https://h.example/", "title": "H"},
    ])
    urls = {r[0] for r in harvest_tool_citations(payload)}
    assert urls == {"https://l.example/", "https://h.example/"}


def test_json_urls_in_plain_string_values_not_harvested():
    payload = json.dumps({"note": "see https://not-a-source.example/page"})
    assert harvest_tool_citations(payload) == []


def test_json_depth_capped():
    inner = {"url": "https://deep.example/"}
    for _ in range(12):
        inner = {"wrap": inner}
    assert harvest_tool_citations(json.dumps(inner)) == []


def test_json_node_count_capped():
    payload = json.dumps({"items": [{"filler": i} for i in range(5000)]
                          + [{"url": "https://tail.example/"}]})
    results = harvest_tool_citations(payload)
    assert len(results) <= 15


def test_json_huge_array_within_cap_bounded():
    import time

    payload = json.dumps([0] * 300000)
    assert len(payload) < 1000000
    started = time.perf_counter()
    assert harvest_tool_citations(payload) == []
    assert time.perf_counter() - started < 2.0


def test_hostile_schemes_rejected():
    payload = json.dumps({"results": [
        {"url": "javascript:alert(1)"},
        {"url": "data:text/html,x"},
        {"url": "file:///etc/passwd"},
        {"url": "ftp://ftp.example/"},
        {"url": "//protocol-relative.example/"},
    ]})
    assert harvest_tool_citations(payload) == []


def test_url_with_control_chars_rejected_not_stripped():
    payload = json.dumps({"url": "http://good.example\n@evil.example"})
    assert harvest_tool_citations(payload) == []


def test_non_str_and_oversized_urls_rejected():
    payload = json.dumps({"results": [
        {"url": 123},
        {"url": None},
        {"url": "https://long.example/" + "a" * 3000},
    ]})
    assert harvest_tool_citations(payload) == []


def test_oversized_input_skipped():
    big = json.dumps({"url": "https://a.example/"}) + " " * 1100000
    assert harvest_tool_citations(big) == []


def test_malformed_and_empty_inputs_silent():
    assert harvest_tool_citations("{not json") == []
    assert harvest_tool_citations("") == []
    assert harvest_tool_citations(None) == []
    assert harvest_tool_citations(12345) == []


def test_no_url_json_yields_no_source():
    assert harvest_tool_citations(json.dumps({"ok": True})) == []


def test_same_url_deduped_within_result():
    payload = json.dumps({"results": [
        {"url": "https://dup.example/", "title": "A"},
        {"url": "https://dup.example/", "title": "B"},
    ]})
    assert len(harvest_tool_citations(payload)) == 1


def test_lone_surrogates_sanitized_in_fields():
    payload = json.dumps({"url": "https://s.example/", "title": "ok"})
    parsed = json.loads(payload)
    parsed["title"] = "bad\ud800title"
    results = harvest_tool_citations(json.dumps(parsed, ensure_ascii=True))
    assert len(results) == 1
    assert "\ud800" not in results[0][1]
    results[0][1].encode("utf-8")
