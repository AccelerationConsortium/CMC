from opentrons import protocol_api

metadata = {
    "protocolName": "CMC Project_V0",
    "description": "Written on 2025.03.02",
    "author": "Zeqing Bao"
}

requirements = {"robotType": "Flex", "apiLevel": "2.19"}


def run(protocol: protocol_api.ProtocolContext):


    # robot setup
    # load 1000 uL tip rack in deck slot D2
    tip1000 = protocol.load_labware(load_name="opentrons_flex_96_filtertiprack_1000ul", location="B1")
    tip50 = protocol.load_labware(load_name="opentrons_flex_96_filtertiprack_50ul", location="B2")
    
    # attach pipette 
    pipette_low = protocol.load_instrument(instrument_name="flex_1channel_50", mount="right", tip_racks=[tip50])
    pipette_high = protocol.load_instrument(instrument_name="flex_1channel_1000", mount="left", tip_racks=[tip1000])

    # load well plate in deck slot D1
    plate = protocol.load_labware(load_name="corning_96_wellplate_360ul_flat", location="D1")

    # load deep well plate in deck slot D2
    deepplate = protocol.load_labware('allenlabresevoir_96_wellplate_2200ul', location = 'D2')


    # load first stock plate with 8 surfactants in deck slot C1
    surfactant_stock_1 = protocol.load_labware(load_name="allenlab_8_wellplate_20000ul", location="C1")
    s1 = surfactant_stock_1['A1']
    s2 = surfactant_stock_1['A2']
    s3 = surfactant_stock_1['A3']
    s4 = surfactant_stock_1['A4']
    s5 = surfactant_stock_1['B1']
    s6 = surfactant_stock_1['B2']
    s7 = surfactant_stock_1['B3']
    s8 = surfactant_stock_1['B4']


    # load second stock plate with 4 surfactants + pyrene in deck slot C2
    surfactant_pyrene_stock_2 = protocol.load_labware(load_name="allenlab_8_wellplate_20000ul", location="C2")
    s9 = surfactant_pyrene_stock_2['A1']
    s10 = surfactant_pyrene_stock_2['A2']
    s11 = surfactant_pyrene_stock_2['A3']
    s12 = surfactant_pyrene_stock_2['A4']
    pyrene = surfactant_pyrene_stock_2['B1']

    # load water in deck slot C3
    water_res = protocol.load_labware('nest_1_reservoir_290ml','C3')
    water = water_res['A1']

    # trash bin
    trash = protocol.load_trash_bin(location="A3")

    sources = {
        's1': s1,
        's2': s2,
        's3': s3,
        's4': s4,
        's5': s5,
        's6': s6,
        's7': s7,
        's8': s8,
        's9': s9,
        's10': s10,
        's11': s11,
        's12': s12,
        'water': water,
        'pyrene': pyrene,
    }
    

    # to be rewritten according to the exp design
################################################################################################################################################

    exp_list = {'new_exp1': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 'new_exp2': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 'new_exp3': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 'new_exp4': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 'new_exp5': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 'new_exp6': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 'new_exp7': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 'new_exp8': {'surfactant_mix_stock_vols': {'s1': 10.909090909090912, 'None_2': 0, 'None_3': 0, 'water': 1789.090909090909}, 'solvent_mix_vol': [0.03333333333333334, 0.047140452079103154, 0.06666666666666665, 0.07500000000000001, 0.085, 0.095, 0.10500000000000001, 0.115, 0.125, 0.15000000000000005, 0.2121320343559643, 0.30000000000000004], 'water_vol': [264.33, 250.8, 231.66, 223.49, 213.69, 203.89, 194.09, 184.29, 174.49, 149.98, 89.09, 2.97], 'pyrene_vol': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}}

    solvent_mix_loc = 'B1'
    row = 'B'

################################################################################################################################################


    def CMC_surfactant_mix(exp, solvent_mix_loc):
        for key in exp['surfactant_mix_stock_vols'].keys():
            vol = exp['surfactant_mix_stock_vols'][key]
            
            if 0 < vol < 50:
                pipette = pipette_low
            else:
                pipette = pipette_high

            if vol > 0:
                pipette.pick_up_tip()
                pipette.transfer(
                    vol,
                    sources[key],
                    deepplate[solvent_mix_loc],
                    new_tip='never',
                    air_gap=vol / 20,
                    blow_out=True
                )
                pipette.drop_tip()



    # make surfactant dilution


        
    def process_a_row(row, vols, source, destination, touch_tip):

        def pipette_selection (vol):
            if vol <= 50:
                return pipette_low
            else:
                return pipette_high
    
        last_pipette = None
        for i, vol in enumerate(vols):
            new_pipette = pipette_selection(vol)

            if new_pipette != last_pipette:
                if last_pipette and last_pipette.has_tip:
                    last_pipette.drop_tip()
                new_pipette.pick_up_tip()

            if vol>0:
                new_pipette.flow_rate.aspirate = vol
                new_pipette.flow_rate.dispense = vol
                new_pipette.flow_rate.blow_out = vol * 5


            transfer_repeat = int((vol // 200) + 1)

            for _ in range(transfer_repeat):
                new_pipette.transfer(vol/transfer_repeat, source, destination[row + str(i+1)], new_tip='never', air_gap=vol/10)

            if touch_tip:
                new_pipette.touch_tip(destination[row + str(i+1)])

            last_pipette = new_pipette

        if last_pipette and last_pipette.has_tip:
            last_pipette.drop_tip()


    def CMC (exp, row, solvent_mix_loc):
        process_a_row(row, exp['water_vol'], water, plate, touch_tip=0)
        process_a_row(row, exp['pyrene_vol'], pyrene, plate, touch_tip=1)
        process_a_row(row, exp['solvent_mix_vol'], deepplate[solvent_mix_loc], plate, touch_tip=1)


    def next_well(well):
        import re
        match = re.match(r"([A-H])(\d+)", well)
        if not match:
            raise ValueError(f"Invalid well format: {well}")
        
        row, col = match.groups()
        col = int(col)

        if col < 12:
            col += 1
        else:
            col = 1
            if row == 'H':
                raise ValueError("Plate overflow: no more wells after H12")
            row = chr(ord(row) + 1)
        return f"{row}{col}"





    # Run the exps
    for exp_key in exp_list.keys():
        exp = exp_list[exp_key]

        CMC_surfactant_mix(exp, solvent_mix_loc)
        CMC(exp, row, solvent_mix_loc)

        solvent_mix_loc = next_well(solvent_mix_loc)
        row = chr(ord(row) + 1)

