#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <arrow/api.h>
#include <plasma/client.h>
#include <plasma/common.h>
#include <arrow/util/logging.h>
#include <arrow/tensor.h>
#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/buffer.h>
#include <arrow/type_fwd.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/datum.h>
#include <arrow/compute/api_scalar.h>

#include <boost/date_time/posix_time/posix_time.hpp>

typedef std::vector<std::shared_ptr<arrow::Table>> TableVector;

TableVector read_tables(plasma::PlasmaClient& client) {
    plasma::ObjectTable objTable;
    ARROW_CHECK_OK(client.List(&objTable));
    std::vector<plasma::ObjectID> obj_ids(objTable.size());
    std::size_t i{0};
    for (auto const& [obj_id, entry] : objTable)
        obj_ids[i++] = obj_id;
    std::vector<plasma::ObjectBuffer> buffers;
    ARROW_CHECK_OK(client.Get(obj_ids, -1, &buffers));
    std::vector<arrow::io::BufferReader> buffer_readers;
    TableVector tables;
    for(const auto& buffer : buffers) {
        auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer.data);
        auto result = arrow::ipc::RecordBatchStreamReader::Open(buffer_reader);
        if(result.ok()) {
            auto stream_reader = result.ValueOrDie();
            auto result = arrow::Table::FromRecordBatchReader(&*stream_reader);
            if(result.ok()) {
                auto table = result.ValueOrDie();
                tables.emplace_back(table);
            }
            else {
                ARROW_LOG(ERROR) << result.status();
            }
        }
        else {
            ARROW_LOG(ERROR) << result.status();
        }
    }
    return tables;
}

void read_vectors(const std::shared_ptr<arrow::Table>& table) {
    uint8_t time_resolution{1}, height_resolution{5};
    int min_height{200}, max_height{800};
    auto datetime = table->GetColumnByName(std::string("datetime"));
    auto l = datetime->length();
    {
        auto result = datetime->GetScalar(l/2);
        if(result.ok()) {
            auto date = result.ValueOrDie();
            std::cout << "type: " << date->type->name() << std::endl;
            std::cout << "value: " << date->ToString() << std::endl;
            std::cout << "year: " << arrow::compute::Year(arrow::Datum(date)).ValueOrDie().scalar()->ToString() << std::endl;

        }
        else {
            ARROW_LOG(ERROR) << result.status();
        }
    }
}

int main() {
    plasma::PlasmaClient client;
    ARROW_CHECK_OK(client.Connect("/tmp/plasma", "", 0));
    auto tables = read_tables(client);
    for(const auto& table: tables)
        for(const auto& field : table->fields())
            std::cout << field->ToString() << std::endl;
    for(const auto& table : tables)
        read_vectors(table);

    ARROW_CHECK_OK(client.Disconnect());
}


















// #include <arrow/api.h>
// #include <plasma/client.h>
// #include <arrow/util/logging.h>
// #include <arrow/tensor.h>
// #include <arrow/array.h>
// #include <arrow/buffer.h>
// #include <arrow/io/memory.h>
// #include <arrow/ipc/writer.h>
// using namespace plasma;
// using namespace arrow;

// int main(int argc, char** argv) {
//   // Start up and connect a Plasma client.
//   PlasmaClient client;
//   ARROW_CHECK_OK(client.Connect("/tmp/plasma", "", 0));
//   // Generate an Object ID for Plasma
//   ObjectID object_id = ObjectID::from_binary("11111111111111111111");

//   // Generate Float Array
//   int64_t input_length = 1000;
//   std::vector<float> input(input_length);
//   for (int64_t i = 0; i < input_length; ++i) {
//     input[i] = 2.0;
//   }

//   // Create Arrow Tensor Object, no copy made!
//   // {input_length} is the shape of the tensor
//   auto value_buffer = Buffer::Wrap<float>(input);
//   Tensor t(float32(), value_buffer, {input_length});

//   // Get the size of the tensor to be stored in Plasma
//   int64_t datasize;
//   ARROW_CHECK_OK(ipc::GetTensorSize(t, &datasize));
//   int32_t meta_len = 0;

//   // Create the Plasma Object
//   // Plasma is responsible for initializing and resizing the buffer
//   // This buffer will contain the _serialized_ tensor
//   std::shared_ptr<Buffer> buffer;
//   ARROW_CHECK_OK(
//       client.Create(object_id, datasize, NULL, 0, &buffer));

//   // Writing Process, this will copy the tensor into Plasma
//   io::FixedSizeBufferWriter stream(buffer);
//   ARROW_CHECK_OK(arrow::ipc::WriteTensor(t, &stream, &meta_len, &datasize));

//   // Seal Plasma Object
//   // This computes a hash of the object data by default
//   ARROW_CHECK_OK(client.Seal(object_id));
// }