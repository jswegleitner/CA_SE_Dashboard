"""Geographic name/code mappings for state and county choropleth maps."""
import re

import pandas as pd


US_STATE_TO_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District Of Columbia': 'DC', 'District of Columbia': 'DC',
}


CA_COUNTY_FIPS = {
    'Alameda': '06001', 'Alpine': '06003', 'Amador': '06005', 'Butte': '06007',
    'Calaveras': '06009', 'Colusa': '06011', 'Contra Costa': '06013', 'Del Norte': '06015',
    'El Dorado': '06017', 'Fresno': '06019', 'Glenn': '06021', 'Humboldt': '06023',
    'Imperial': '06025', 'Inyo': '06027', 'Kern': '06029', 'Kings': '06031',
    'Lake': '06033', 'Lassen': '06035', 'Los Angeles': '06037', 'Madera': '06039',
    'Marin': '06041', 'Mariposa': '06043', 'Mendocino': '06045', 'Merced': '06047',
    'Modoc': '06049', 'Mono': '06051', 'Monterey': '06053', 'Napa': '06055',
    'Nevada': '06057', 'Orange': '06059', 'Placer': '06061', 'Plumas': '06063',
    'Riverside': '06065', 'Sacramento': '06067', 'San Benito': '06069', 'San Bernardino': '06071',
    'San Diego': '06073', 'San Francisco': '06075', 'San Joaquin': '06077', 'San Luis Obispo': '06079',
    'San Mateo': '06081', 'Santa Barbara': '06083', 'Santa Clara': '06085', 'Santa Cruz': '06087',
    'Shasta': '06089', 'Sierra': '06091', 'Siskiyou': '06093', 'Solano': '06095',
    'Sonoma': '06097', 'Stanislaus': '06099', 'Sutter': '06101', 'Tehama': '06103',
    'Trinity': '06105', 'Tulare': '06107', 'Tuolumne': '06109', 'Ventura': '06111',
    'Yolo': '06113', 'Yuba': '06115',
}


def to_state_code(x):
    """Resolve a state name or 2-letter code to a USPS abbreviation, or None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if len(s) == 2 and s.upper() in US_STATE_TO_ABBREV.values():
        return s.upper()
    key = s.title()
    if key in US_STATE_TO_ABBREV:
        return US_STATE_TO_ABBREV[key]
    if s.upper() in US_STATE_TO_ABBREV.values():
        return s.upper()
    return None


def to_county_fips(x):
    """Resolve a California county name (with optional ' County' suffix) to a FIPS code, or None."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    key = s.title()
    if key in CA_COUNTY_FIPS:
        return CA_COUNTY_FIPS[key]
    cleaned = re.sub(r'\s+County$', '', key, flags=re.IGNORECASE).title()
    if cleaned in CA_COUNTY_FIPS:
        return CA_COUNTY_FIPS[cleaned]
    return None
