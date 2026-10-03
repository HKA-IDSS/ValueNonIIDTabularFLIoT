import os

from Definitions import ROOT_DIR
from util.manual_partition_scripts.EdgeIIOTCoreset.MaverickDDOSUDP import partition_edgeiiot_coreset_1_Maverick_ddos_udp
from util.manual_partition_scripts.EdgeIIOTCoreset.MaverickLeastClass import \
    partition_edgeiiot_coreset_1_Maverick_least_class
from util.manual_partition_scripts.EdgeIIOTCoreset.MaverickOnlyNormal import \
    partition_edgeiiot_coreset_1_Maverick_only_normal
from util.manual_partition_scripts.EdgeIIOTCoreset.MaverickSQLInjection import \
    partition_edgeiiot_coreset_1_Maverick_sql_injection
from util.manual_partition_scripts.ElectricConsumption.AbstractMaverickPartition import \
    partition_maverick_categorical_feature
from util.manual_partition_scripts.ElectricConsumption.FeatureSkewBuildingType import feature_skew_building_type
from util.manual_partition_scripts.ElectricConsumption.FeatureSkewFacilityType import feature_skew_facility_type
from util.manual_partition_scripts.ElectricConsumption.FeatureSkewStateFactor import feature_skew_state_factor
from util.manual_partition_scripts.ElectricConsumption.NonIIDSampling import wids_energy_non_iid_by_label
from util.manual_partition_scripts.ElectricConsumption.RandomSampling import wids_energy_iid_sampling
from util.manual_partition_scripts.HAR.HAR_1_Maverick_1_MissingOneLabel import \
    partition_har_1_maverick_1_missingonelabel
from util.manual_partition_scripts.HAR.HAR_1_Maverick_1_MissingTwoLabels import \
    partition_har_1_maverick_1_missingtwolabels
from util.manual_partition_scripts.HAR.HAR_1_Maverick_Balanced_Laying import partition_har_1_maverick_laying_balanced
from util.manual_partition_scripts.HAR.HAR_1_Maverick_Balanced_WalkingUpstairs import \
    partition_har_1_maverick_walkingupstairs_balanced
from util.manual_partition_scripts.HAR.HAR_1_Maverick_Laying import partition_har_1_maverick_laying
from util.manual_partition_scripts.HAR.HAR_1_Maverick_WalkingUpstairs import partition_har_1_maverick_walkingupstairs

directory_for_data = ROOT_DIR + os.sep + "data" + os.sep + "partitioned_training_data"

manual_partitions = {
    "HAR_1_Maverick_Laying": partition_har_1_maverick_laying,
    "HAR_1_Maverick_WalkingUpstairs": partition_har_1_maverick_walkingupstairs,
    "HAR_1_Maverick_Laying_Balanced": partition_har_1_maverick_laying_balanced,
    "HAR_1_Maverick_WalkingUpstairs_Balanced": partition_har_1_maverick_walkingupstairs_balanced,
    "HAR_1_Maverick_1_MissingOneLabel": partition_har_1_maverick_1_missingonelabel,
    "HAR_1_Maverick_1_MissingTwoLabels": partition_har_1_maverick_1_missingtwolabels,
    "edgeiot_coreset_1_Maverick_Least_Class": partition_edgeiiot_coreset_1_Maverick_least_class,
    "edgeiot_coreset_1_Maverick_Only_Normal": partition_edgeiiot_coreset_1_Maverick_only_normal,
    "edgeiot_coreset_1_Maverick_sql_injection": partition_edgeiiot_coreset_1_Maverick_sql_injection,
    "edgeiot_coreset_1_Maverick_ddos_udp": partition_edgeiiot_coreset_1_Maverick_ddos_udp,
    "wids_energy_non_iid_by_label": wids_energy_non_iid_by_label,
    "wids_energy_iid_sampling": wids_energy_iid_sampling,
    "wids_energy_feature_skew_building_type": feature_skew_building_type,
    "wids_energy_feature_skew_state_factor": feature_skew_state_factor,
    "wids_energy_feature_skew_facility_type": feature_skew_facility_type,
    "wids_energy_maverick_facility_type_grocery_store": partition_maverick_categorical_feature(
        "facility_type", "Grocery_store_or_food_market",
        partition_name_prefix="wids_energy_maverick_facility_type_grocery_store"
    ),
    "wids_energy_maverick_facility_type_uncategorized_multifamily": partition_maverick_categorical_feature(
        "facility_type", "Multifamily_Uncategorized",
        partition_name_prefix="wids_energy_maverick_facility_type_uncategorized_multifamily"
    ),
    # "Wine_Maverick": electric_consumption_non_iid_sampling
}


def sample_data_manual_partition(partition_name, seed):
    if not os.path.exists(directory_for_data +
                          os.sep + "manual" +
                          os.sep + partition_name +
                          os.sep + str(seed)):
        manual_partitions[partition_name](seed)
