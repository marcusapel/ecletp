# config_etp.py

import os
import json
from urllib import request, parse
from dataclasses import dataclass
from typing import Optional

STATIC_REFRESH_TOKEN = "1.AQIANaKkOuK21UiRlX_PBbRZsO6_0uu67LdHozwBfQExh50CADgCAA.AgABAwEAAABVrSpeuWamRam2jAF1XRQEAwDs_wUA9P-0dL7gR1WFIIy_-9UehHQq07MWAWYo7hhKDiMGrmZZeVSNvaR_19PIfTamXb3BjedroiJkPM7h0Xot6SjY-BgCMa4pt8ml8m0LJasv5TOAd-zMnCTfohqbBF53EJXmjiIoIY3L-1rDjS2C3muqVenlRS95q_cOzrUOSTXtB6-8cdBs_mVnnucZsJnTO4-2uDIm8zf6_vwTlr43Np1-1X2rLQAn6iSNP1-oAb9irdFsSin1VbFyqy64Mwpf6EjWKl8aGFQGYZBXCtmcxyDMtKG000oVY6RkgI0iD2NXgLOfyeySHzFs6FqQxsSQqld4WP387esRjajfABcEREr2iZJZRbrtoUq0nsHCuXS3ZwhcZ-PUSCGX842qIidd4DPjsRdv1i8Pg8apt7XV0s3u4EPD9fIxopB7EZj8KrzDAfpcxNEEvVhDU6Bc4nP1mQ3Sww5duC3Qm2NJoVyQZPHFKiERzG9p1MzIlNMT2la9a1l0zlTj0-Zasov1BW1KbFxyLl4T5fZtxpwKxEqCH19yCrCT91BMcIM8NiNNe2_XM3qHCH-c_rWqE--5zjaqSDArW1Ds2GUZ3q8QMZy5NlEXPcdxT64WB-0ncPBumv-s-Rufo6vqW_NoBHPKOw9QYV7EH0Mw-IMZ8xix6qR7gmPlBBEqcR8qo9zWTZ7wK_q0aL6LUc4h3pLSPzYBPcvFT_p6cRmCMiwnyollZvGjw0b-Zsx9jdAo7gRTEt01_jt3SmsN2LXd8ZlmYS5Oqwg5mMHjQ6Ps2yarz5FIQEKmpoX61mNiLVdAGi2KXTIklRaHulknLjDbxZxpxCE7NRZrQhVfz23PPTKTg_eiOLyl9l956qFk4_objdkK7ESrx5O1UMZxUXLlOeZq7YESHe0VMjLnMQXEH5zIgivhYrBbZLcC4XRLA6HwOmI50tqdLIv0lQs7Ym6bAZfb-qVPQOtUPd_opC1KZo-U1jGxw344pozRtTpTm-jWSNIKEYHJaPDOUrVre8ELRbMOOsKtkjmeJ4SW_T5L3-SyUywJHTX3BJKR2ieHN8BQa0AZxy2wvy3Mr-rpg6cu8Q"

@dataclass
class Config:
    rddms_host: str
    data_partition_id: str
    dataspace_uri: str
    token: Optional[str]
    inline_threshold: int = 0
    compress_over_bytes: int = 1024
    array_path_style: str = 'uuid-path'
    use_protocol9: bool = True
    object_format: str = 'XML'
    epsg: int = 2334  # Default EPSG code

    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            rddms_host=os.getenv('RDDMS_HOST', os.getenv('ETP_URL', 'wss://equinordev.energy.azure.com/api/reservoir-ddms-etp/v2/')),
            data_partition_id=os.getenv('DATA_PARTITION_ID', os.getenv('DATA_PARTITION', 'data')),
            dataspace_uri=os.getenv('DATASPACE', "eml:///dataspace('maap/m25test')"),
            token=get_token(),
        )

def get_token() -> str:
    url = "https://login.microsoftonline.com/3aa4a235-b6e2-48d5-9195-7fcf05b459b0/oauth2/v2.0/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {
        "grant_type": "refresh_token",
        "client_id": "ebd2bfee-ecba-47b7-a33c-017d0131879d",
        "scope": "7daee810-3f78-40c4-84c2-7a199428de18/.default openid offline_access",
        "refresh_token": STATIC_REFRESH_TOKEN  
    }
    try:
        payload = json.loads(
            request.urlopen(
                request.Request(url, data=parse.urlencode(data).encode("utf-8"), headers=headers, method="POST"),
                timeout=60
            ).read().decode("utf-8")
        )
        token = payload.get("access_token")
        if not token:
            raise RuntimeError(payload.get("error_description") or payload.get("error") or "No access_token in response")
        return token
    except Exception as e:
        raise RuntimeError(f"Token request failed: {e}")


