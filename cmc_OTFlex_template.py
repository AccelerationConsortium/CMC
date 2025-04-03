from opentrons import protocol_api

metadata = {
    "protocolName": "CMC Project_V0",
    "description": "Written on 2025.03.02",
    "author": "Zeqing Bao"
}

requirements = {"robotType": "Flex", "apiLevel": "2.19"}


def run(protocol: protocol_api.ProtocolContext):


    # load 1000 uL tip rack in deck slot D2
    tip1000 = protocol.load_labware(load_name="opentrons_flex_96_filtertiprack_1000ul", location="B1")
    tip50 = protocol.load_labware(load_name="opentrons_flex_96_filtertiprack_50ul", location="B2")
    
    # attach pipette to right mount
    pipette = protocol.load_instrument(instrument_name="flex_1channel_1000", mount="left", tip_racks=[tip1000, tip50])


    # load well plate in deck slot B1
    plate = protocol.load_labware(load_name="corning_96_wellplate_360ul_flat", location="D1")

    # load deep well plate in deck slot B3
    deepplate = protocol.load_labware(load_name="corning_96_wellplate_360ul_flat", location="D2")

    # load vial tray in deck slot A3
    stocks = protocol.load_labware(load_name="allenlab_8_wellplate_20000ul", location="C1")
    water_loc = stocks['A1']
    surfactant_loc = stocks['A2']

    probe_methanol_loc = stocks['B1']
    probe_DMSO_loc = stocks['B2']
    probe_ethanol_loc = stocks['B3']

    trash = protocol.load_trash_bin(location="A3")


    probe_vol = [10] * 12
    surfactant_vol = [0, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 500]
    water_vol = [990, 980, 970, 940, 890, 840, 790, 740, 690, 640, 590, 490]

    # Set up pipette aspirating and dispensing flow rate
    pipette.flow_rate.aspirate = 300
    pipette.flow_rate.dispense = 300
    pipette.flow_rate.aspirate = 300
    pipette.flow_rate.dispense = 300
    pipette.flow_rate.blow_out = 500
    pipette.flow_rate.blow_out = 500

    def CMC (probe, row):

        pipette.well_bottom_clearance.dispense = 30
        pipette.well_bottom_clearance.aspirate = 3

        if probe == "methanol":
            probe_loc = probe_methanol_loc
        
        elif probe == "DMSO":
            probe_loc = probe_DMSO_loc

        elif probe == "ethanol":
            probe_loc = probe_ethanol_loc

        pipette.pick_up_tip(tip1000)
        for col in range(1, 13):
            pipette.transfer(water_vol[col-1], water_loc, deepplate[row + str(col)], new_tip='never', blow_out=True)
        pipette.drop_tip()

        pipette.pick_up_tip(tip1000)
        for col in range(1, 13):
            pipette.transfer(surfactant_vol[col-1], surfactant_loc, deepplate[row + str(col)], new_tip='never', blow_out=True)
        pipette.drop_tip()

        pipette.pick_up_tip(tip50)
        for col in range(1, 13):
            pipette.transfer(probe_vol[col-1], probe_loc, deepplate[row + str(col)], new_tip='never', blow_out=True)
        pipette.drop_tip()


        pipette.well_bottom_clearance.dispense = 3
        pipette.well_bottom_clearance.aspirate = 3


        pipette.pick_up_tip(tip1000)
        for col in range(1, 13):
            pipette.mix(3, 200, deepplate[row + str(col)])
            pipette.transfer(200, deepplate[row + str(col)], plate[row + str(col)], new_tip='never', blow_out=True)
        pipette.drop_tip()

    CMC("DMSO", 'A')
    CMC("ethanol", 'B')
    CMC("methanol", 'C')