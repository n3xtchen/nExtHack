#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""


from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(latitude:float, longitude:float):
    """Get current temperature for provided coordinates in celsius.

    Args:
        latitude: first number
        longitude: second number
    """
    # response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    # data = response.json()
    # return data['current']['temperature_2m']
    return 22.2

if __name__ == "__main__":
    mcp.run(transport="stdio")
