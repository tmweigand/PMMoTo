# distutils: language = c++
# cython: profile=True
# cython: linetrace=True


def _single_merge_matched_sets(all_set_data):
    """
    Connect all matched sets from entire domain
    """

    # Convert list of lists to list and create keys
    matches = []
    all_matches = {}
    for proc_match in all_set_data:
        for _,match in proc_match['matched_sets'].items():
            matches.append(match)
            all_matches[match.ID] = match

    # Loop through all matched sets
    merged_sets = 0
    for match in matches:
        if not match.visited:
            match.visited = True
            queue = [match.ID]
            connections = [match.ID]
            inlet = match.inlet
            outlet = match.outlet

            while len(queue) > 0:
            
                current_match = all_matches[queue.pop()]
            
                if not inlet:
                    inlet = current_match.inlet
                if not outlet:
                    outlet = current_match.outlet
            
                for n_ID in current_match.n_ID:
                    if not all_matches[n_ID].visited:
                        all_matches[n_ID].visited = True
                        queue.append(n_ID)
                        connections.append(n_ID)

            for connect in connections:
                all_matches[connect].n_ID = connections
                all_matches[connect].inlet = inlet
                all_matches[connect].outlet = outlet
                all_matches[connect].global_ID = merged_sets

            if connections:
                        merged_sets += 1

    return matches,merged_sets
